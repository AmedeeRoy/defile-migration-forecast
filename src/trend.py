"""A cautious annual trend correction, from `data/count/species_year_statistics.json`.

Built by `scripts/build_trend_stats.py`: one record per species, a Holt damped-trend fit
of the effort-corrected annual seasonal rate (birds/observer-hour, summed over the trained
season, 1993 onward -- see `data/count/readme.md`, real daily monitoring only started
then). Rolling-origin one-year-ahead backtesting (see DECISIONS.md) showed this beats both
a no-trend baseline and an unconstrained linear-trend extrapolation on mean skill and,
more importantly, on worst-case downside: species with no real multi-year trend (Common
Buzzard, Marsh Harrier, Osprey) are barely hurt, while species with a real one (Red Kite,
Kestrel, Black Kite) get most of the available benefit.

This is applied as a scalar multiplier on the trained model's output for the forecast
year -- not fed into the network as an input. ~30 sparse annual points is not enough for a
deep net to learn a safe extrapolation (an unconstrained linear fit already demonstrated
that failure mode in the backtest); keeping the correction as an auditable post-hoc factor
means it can be inspected, logged, and disabled independently of the model itself.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

TREND_FILE = os.path.join("count", "species_year_statistics.json")

# Fixed damping hyperparameters, chosen a priori (standard Holt-damped-trend defaults) and
# not fitted per species: with ~30 annual points, tuning alpha/beta/phi per species on the
# same data used to evaluate them would reintroduce the overfitting this method exists to
# avoid. A pooled grid search over the same rolling-origin protocol (36 combos, DECISIONS.md)
# confirmed these are not a fragile cherry-pick: they rank top-5, within 0.7% RMSE of the
# best combo found, and the worst-performing region of the grid is exactly where theory
# predicts (phi=0.95, i.e. barely damped, paired with slow adaptation) -- confined there,
# and only 1.47x worse than the best at its most extreme.
ALPHA = 0.4
BETA = 0.15
PHI = 0.7

# Hard sanity bound on the multiplier, symmetric in log space: caps it to [1/3, 3]
# regardless of what the fit implies. This exists because of a real blind spot the backtest
# cannot close: a species whose entire fitted history moves in one direction (Red Kite:
# zero sign changes in its rolling 7-year trend across all 33 years) has no historical test
# case for what happens when that trend eventually plateaus or reverses -- the backtest can
# only certify "rides an ongoing trend well", not "handles a turning point it has never
# seen". The clamp is a backstop against a bad fit or a genuine future reversal silently
# producing an unbounded correction; the warning makes it visible rather than silent, per
# this project's "a silently wrong forecast is as serious as a crash" doctrine.
MAX_LOG_MULTIPLIER = np.log(3.0)


@dataclass
class Trend:
    """Damped-trend annual correction for one species.

    `level` and `trend` are the Holt state (log-rate scale) at `last_year`; `level_path` is
    the fitted level for every year in `year`, kept for diagnostics/plotting; the raw
    observed `log_rate` per year is kept alongside it for the same reason.
    """

    species: str
    year: np.ndarray  # (Y,)
    log_rate: np.ndarray  # (Y,) observed effort-corrected annual log-rate
    level_path: np.ndarray  # (Y,) fitted Holt level, same length/order as `year`
    baseline_log_rate: float  # mean of level_path; the multiplier's reference point
    level: float  # final Holt level at last_year
    trend: float  # final Holt trend increment at last_year
    phi: float
    last_year: int

    @classmethod
    def load(cls, data_dir: str, species: str) -> "Trend":
        path = os.path.join(data_dir, TREND_FILE)
        with open(path) as f:
            entries = json.load(f)

        for entry in entries:
            if entry["species"] == species:
                return cls(
                    species=species,
                    year=np.asarray(entry["year"], dtype=int),
                    log_rate=np.asarray(entry["log_rate"], dtype=float),
                    level_path=np.asarray(entry["level_path"], dtype=float),
                    baseline_log_rate=float(entry["baseline_log_rate"]),
                    level=float(entry["level"]),
                    trend=float(entry["trend"]),
                    phi=float(entry["phi"]),
                    last_year=int(entry["last_year"]),
                )

        available = ", ".join(sorted(e["species"] for e in entries))
        raise KeyError(f"No trend stats for species {species!r} in {path}. Available: {available}")

    def forecast_log_rate(self, year: int) -> float:
        """Damped-trend forecast of the annual log-rate for `year`.

        For `year` beyond `last_year`, Holt's damped multi-step formula sums a
        geometrically-decaying trend increment (`phi + phi^2 + ... + phi^h`) rather than
        continuing the raw slope `h` times -- the mechanism that keeps a multi-year-ahead
        call from running away. In normal use `year == last_year + 1`, since the file is
        rebuilt every season; larger `h` only arises if the file goes stale.

        For `year` at or before `last_year` (re-scoring a past season), returns that
        year's own fitted level rather than extrapolating.
        """
        h = year - self.last_year
        if h <= 0:
            idx = int(np.clip(np.searchsorted(self.year, year), 0, len(self.year) - 1))
            return float(self.level_path[idx])

        damped_sum = h if self.phi == 1 else self.phi * (1 - self.phi**h) / (1 - self.phi)
        return self.level + damped_sum * self.trend

    def multiplier(self, year: int) -> float:
        """Scalar correction factor for the trained model's output for `year`.

        Centred on `baseline_log_rate` (the mean of the fitted level over the years used
        to build this file) rather than on zero: a model trained with `year_used:
        "constant"` sees no year signal at all, so it has implicitly learned something
        like the average annual level over its training years. This multiplier expresses
        how far `year`'s expected level is from that average, so it composes with the
        model's output as a relative adjustment rather than replacing what the model
        already learned.

        Clamped to `[exp(-MAX_LOG_MULTIPLIER), exp(MAX_LOG_MULTIPLIER)]` -- see that
        constant's docstring for why. Logs a warning when the clamp actually binds, since
        that means this species' correction is being held back from what the fit itself
        implies.
        """
        log_mult = self.forecast_log_rate(year) - self.baseline_log_rate
        clamped = float(np.clip(log_mult, -MAX_LOG_MULTIPLIER, MAX_LOG_MULTIPLIER))
        if clamped != log_mult:
            log.warning(
                f"{self.species}: trend multiplier for {year} clamped from "
                f"{np.exp(log_mult):.3f}x to {np.exp(clamped):.3f}x (see MAX_LOG_MULTIPLIER)."
            )
        return float(np.exp(clamped))
