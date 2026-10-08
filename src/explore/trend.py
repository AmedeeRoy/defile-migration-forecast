"""Trend models for the Explore page: one taxon's smooth trend, season and phenology shift, and
gap-filled annual totals. Part of `src.explore` (see its `__init__` for the boundary).

One model, two priors, benchmarked by `scripts/benchmark_trend.py` before either is exported:

    count_day ~ NegBin(mean = c_day * exp(eta), theta)
    eta = a + trend(year) + year_effect(year) + season(doy) + shift(year, doy)

`c_day` is the day's coverage (`src.explore.profile.coverage`: the share of the day's expected
passage in the hours counted), `year_effect` an independent level per year (good and bad years),
`shift` a smooth change of the season's timing or width over the years. The negative binomial is a
day's gamma-distributed rate seen through a Poisson count, so the hours not counted share the day's
over-dispersion with the hours counted (`fill_draws`).

Every component is a basis times coefficients with a Gaussian prior, so both variants are fitted
the same way: the posterior mode by penalised IRLS, and the prior's hyperparameters and `theta` by
the Laplace approximation of the marginal likelihood (`fit`). They differ only in the prior:

- `gam`: P-splines, cubic B-spline bases with second-difference penalties (a tensor product for
  `shift`), a smoothing parameter per penalty, as mgcv would.
- `gp`: Gaussian processes with Matern-5/2 kernels (a product of two for `shift`) plus a linear
  year term, in the Hilbert-space basis-function approximation (Solin and Sarkka 2020): a variance
  and lengthscale per kernel. Beyond the data it reverts to the linear trend, where the P-spline
  extrapolates its last slope.
- `season`: the season alone, a fixed phenology without trend or year effect, as the forecast's
  baseline has: the reference both must beat.

Uncertainty is the posterior given the fitted hyperparameters (empirical Bayes), and days are
independent given the model, so a run of bad-weather days is not: intervals for annual totals
are likely too narrow where many days are filled. The benchmark measures it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import BSpline
from scipy.special import gammaln

from src.explore.export import in_window

VARIANTS = ("season", "gam", "gp")

# Prior precision on coefficients no penalty reaches (intercept, spline null spaces): effectively
# flat on the log scale (sd 100), and what keeps the overlapping constants identifiable.
RIDGE = 1e-4
ETA_MAX = 20.0  # log birds per full-day equivalent; guards exp() in early IRLS steps

# P-spline bases (cubic, second-order difference penalties).
GAM_TREND_K = 12
GAM_SEASON_K = 20
GAM_SHIFT_K = (6, 8)  # year x doy
GAM_LOG_LAMBDA_BOUNDS = (-6.0, 16.0)

# Hilbert-space GP: basis functions per dimension, and the domain's half-width as a multiple of
# the data's (>= 1.5 keeps the approximation accurate at the edges, Riutort-Mayol et al. 2023).
GP_TREND_M = 20
GP_SEASON_M = 25
GP_SHIFT_M = (8, 10)
GP_BOUNDARY = 1.5
GP_SIGMA_BOUNDS = (0.01, 10.0)
GP_LENGTHSCALE_BOUNDS = {  # years / days
    "trend": (2.0, 60.0),
    "season": (4.0, 150.0),
    "shift_year": (3.0, 60.0),
    "shift_doy": (6.0, 150.0),
}
THETA_BOUNDS = (0.02, 200.0)  # negative binomial shape; small = over-dispersed

# Below this coverage a day says nothing about its rate (75 Black Kites in 11 minutes on
# 2025-07-27, c = 0.0004, would be a 190 000-bird day): it is neither fitted nor conditioned on,
# and its birds are added to the filled total as they are (`extra`).
MIN_COVERAGE = 0.1

IRLS_MAX_ITER = 100
IRLS_TOL = 1e-8
DRAWS = 1000
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)


# --- data -------------------------------------------------------------------


def model_frame(
    days: pd.DataFrame, effort: pd.DataFrame, c: pd.Series, first_year: int, last_year: int
) -> pd.DataFrame:
    """One row per window day of `first_year`..`last_year`: `date`, `year`, `doy`, `c`, `y` and
    `extra`.

    Counted days (`state == "counted"`) carry their coverage `c` and count `y` (0 without a record);
    every other day, and a counted day with only a presence record, has `c = 0` and `y = 0`: nothing
    observed, all of it to fill. A counted day below `MIN_COVERAGE` is treated as not counted, its
    birds kept in `extra`.
    """
    dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
    dates = dates[in_window(pd.Series(dates)).to_numpy()]
    f = pd.DataFrame({"date": dates, "year": dates.year, "doy": dates.dayofyear})
    counted = effort["state"] == "counted"
    f["c"] = f["date"].map(c[counted].set_axis(effort.loc[counted, "date"])).fillna(0.0)
    f["y"] = f["date"].map(days.set_index("date")["count"])
    no_count = (f["c"] > 0) & f["y"].isna() & f["date"].isin(days["date"])  # presence only
    f.loc[no_count, "c"] = 0.0
    f["y"] = f["y"].where(f["c"] > 0, 0.0).fillna(0.0)
    low = f["c"] < MIN_COVERAGE
    f["extra"] = f["y"].where(low, 0.0)
    f.loc[low, ["c", "y"]] = 0.0
    return f


# --- bases and priors -------------------------------------------------------


def bspline_basis(x: np.ndarray, lo: float, hi: float, k: int) -> np.ndarray:
    """Cubic B-spline basis with `k` functions on equally spaced knots over [lo, hi]."""
    inner = np.linspace(lo, hi, k - 2)
    step = inner[1] - inner[0]
    knots = np.r_[lo - 3 * step + step * np.arange(3), inner, hi + step * np.arange(1, 4)]
    return BSpline.design_matrix(np.clip(x, lo, hi), knots, 3).toarray()


def difference_penalty(k: int, order: int = 2) -> np.ndarray:
    d = np.diff(np.eye(k), order, axis=0)
    return d.T @ d


def hs_basis(x: np.ndarray, half: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert-space basis on [-half, half] (x already centred): `(phi (n, m), sqrt
    eigenvalues)`."""
    j = np.arange(1, m + 1)
    omega = np.pi * j / (2 * half)
    return np.sin(omega * (x[:, None] + half)) / np.sqrt(half), omega


def matern52_spectral(omega: np.ndarray, lengthscale: float) -> np.ndarray:
    """Unit-variance Matern-5/2 spectral density in one dimension."""
    nu = 2.5
    log_s = (
        np.log(2 * np.sqrt(np.pi))
        + gammaln(nu + 0.5)
        - gammaln(nu)
        + nu * np.log(2 * nu)
        - 2 * nu * np.log(lengthscale)
        - (nu + 0.5) * np.log(2 * nu / lengthscale**2 + omega**2)
    )
    return np.exp(log_s)


@dataclass
class Block:
    """A model component: its columns in the design and how its prior precision follows from the
    hyperparameters."""

    name: str
    columns: slice
    precision: callable  # dict of hyperparameters -> (k, k) precision matrix


@dataclass
class Design:
    variant: str
    years: np.ndarray  # every year of the model's range, observed or not
    doy_range: tuple[int, int]
    blocks: list[Block]
    hyper_names: list[str]
    hyper_bounds: list[tuple[float, float]]
    hyper_start: list[float]
    _basis: callable = field(repr=False)

    def X(self, year: np.ndarray, doy: np.ndarray) -> np.ndarray:
        return self._basis(np.asarray(year), np.asarray(doy))

    def precision(self, h: dict) -> np.ndarray:
        n = self.blocks[-1].columns.stop
        Q = np.zeros((n, n))
        for b in self.blocks:
            Q[b.columns, b.columns] = b.precision(h)
        return Q + RIDGE * np.eye(n)

    def columns_of(self, *names: str) -> np.ndarray:
        """Indices of the named blocks' columns."""
        return np.concatenate(
            [np.arange(b.columns.start, b.columns.stop) for b in self.blocks if b.name in names]
            + [np.zeros(0, int)]
        )


def design(variant: str, years: np.ndarray, doy_range: tuple[int, int]) -> Design:
    """The bases and priors of a variant over `years` (all of them, including years without data,
    whose coefficients then stay at their prior) and the window's days of year."""
    y0, y1 = int(years.min()), int(years.max())
    d0, d1 = doy_range
    y_mid, y_half = (y0 + y1) / 2, max((y1 - y0) / 2, 1.0)
    d_mid, d_half = (d0 + d1) / 2, (d1 - d0) / 2
    n_years = y1 - y0 + 1
    parts, blocks, names, bounds, start = [], [], [], [], []
    col = 0

    def add(name, basis, precision):
        nonlocal col
        k = basis(np.array([y0]), np.array([d0])).shape[1]
        parts.append(basis)
        blocks.append(Block(name, slice(col, col + k), precision))
        col += k

    def hyper(name, lo, hi, x0):
        names.append(name)
        bounds.append((lo, hi))
        start.append(x0)

    add("intercept", lambda y, d: np.ones((len(y), 1)), lambda h: np.zeros((1, 1)))
    log_sigma = (np.log(GP_SIGMA_BOUNDS[0]), np.log(GP_SIGMA_BOUNDS[1]))
    lam = GAM_LOG_LAMBDA_BOUNDS

    if variant in ("gam", "season"):
        S = difference_penalty(GAM_SEASON_K)
        add(
            "season",
            lambda y, d: bspline_basis(d, d0, d1, GAM_SEASON_K),
            lambda h: np.exp(h["season"]) * S,
        )
        hyper("season", *lam, 2.0)
    if variant == "gam":
        St = difference_penalty(GAM_TREND_K)
        add(
            "trend",
            lambda y, d: bspline_basis(y, y0, y1, GAM_TREND_K),
            lambda h: np.exp(h["trend"]) * St,
        )
        hyper("trend", *lam, 2.0)
        ky, kd = GAM_SHIFT_K
        Sy, Sd = difference_penalty(ky), difference_penalty(kd)
        add(
            "shift",
            lambda y, d: _row_kron(bspline_basis(y, y0, y1, ky), bspline_basis(d, d0, d1, kd)),
            lambda h: np.exp(h["shift_year"]) * np.kron(Sy, np.eye(kd))
            + np.exp(h["shift_doy"]) * np.kron(np.eye(ky), Sd),
        )
        hyper("shift_year", *lam, 4.0)
        hyper("shift_doy", *lam, 4.0)
        add(
            "year",
            lambda y, d: np.eye(n_years)[np.clip(y - y0, 0, n_years - 1)],
            lambda h: np.exp(h["year"]) * np.eye(n_years),
        )
        hyper("year", *lam, 2.0)
    if variant == "gp":
        yh, dh = GP_BOUNDARY * y_half, GP_BOUNDARY * d_half
        _, om_t = hs_basis(np.zeros(1), yh, GP_TREND_M)
        _, om_s = hs_basis(np.zeros(1), dh, GP_SEASON_M)
        _, om_y = hs_basis(np.zeros(1), yh, GP_SHIFT_M[0])
        _, om_d = hs_basis(np.zeros(1), dh, GP_SHIFT_M[1])
        add(
            "linear",
            lambda y, d: ((y - y_mid) / y_half)[:, None],
            lambda h: np.array([[np.exp(-2 * h["linear"])]]),
        )
        hyper("linear", *log_sigma, np.log(0.5))
        add(
            "trend",
            lambda y, d: hs_basis(y - y_mid, yh, GP_TREND_M)[0],
            lambda h: np.diag(
                1 / (np.exp(2 * h["trend"]) * matern52_spectral(om_t, np.exp(h["trend_ls"])))
            ),
        )
        hyper("trend", *log_sigma, np.log(0.5))
        hyper("trend_ls", *np.log(GP_LENGTHSCALE_BOUNDS["trend"]), np.log(8.0))
        add(
            "season",
            lambda y, d: hs_basis(d - d_mid, dh, GP_SEASON_M)[0],
            lambda h: np.diag(
                1 / (np.exp(2 * h["season"]) * matern52_spectral(om_s, np.exp(h["season_ls"])))
            ),
        )
        hyper("season", *log_sigma, np.log(2.0))
        hyper("season_ls", *np.log(GP_LENGTHSCALE_BOUNDS["season"]), np.log(20.0))
        add(
            "shift",
            lambda y, d: _row_kron(
                hs_basis(y - y_mid, yh, GP_SHIFT_M[0])[0],
                hs_basis(d - d_mid, dh, GP_SHIFT_M[1])[0],
            ),
            lambda h: np.diag(
                1
                / (
                    np.exp(2 * h["shift"])
                    * np.kron(
                        matern52_spectral(om_y, np.exp(h["shift_year_ls"])),
                        matern52_spectral(om_d, np.exp(h["shift_doy_ls"])),
                    )
                )
            ),
        )
        hyper("shift", *log_sigma, np.log(0.3))
        hyper("shift_year_ls", *np.log(GP_LENGTHSCALE_BOUNDS["shift_year"]), np.log(10.0))
        hyper("shift_doy_ls", *np.log(GP_LENGTHSCALE_BOUNDS["shift_doy"]), np.log(30.0))
        add(
            "year",
            lambda y, d: np.eye(n_years)[np.clip(y - y0, 0, n_years - 1)],
            lambda h: np.exp(-2 * h["year"]) * np.eye(n_years),
        )
        hyper("year", *log_sigma, np.log(0.3))
    hyper("theta", *np.log(THETA_BOUNDS), np.log(1.0))

    def basis(y, d):
        return np.hstack([p(y, d) for p in parts])

    return Design(variant, np.arange(y0, y1 + 1), doy_range, blocks, names, bounds, start, basis)


def _row_kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product: the tensor-product basis."""
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


# --- fitting ----------------------------------------------------------------


def nb_loglik(y: np.ndarray, mu: np.ndarray, theta: float) -> np.ndarray:
    return (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1)
        + theta * np.log(theta / (theta + mu))
        + y * np.log(mu / (theta + mu) + 1e-300)
    )


def posterior_mode(X, offset, y, Q, theta, beta):
    """Penalised IRLS (Fisher scoring with step halving) for the negative binomial log link:

    `(beta, H, penalised log-likelihood)`, H the penalised expected information.
    """

    def objective(b):
        eta = np.minimum(offset + X @ b, ETA_MAX)
        return nb_loglik(y, np.exp(eta), theta).sum() - 0.5 * b @ Q @ b

    current = objective(beta)
    for _ in range(IRLS_MAX_ITER):
        mu = np.exp(np.minimum(offset + X @ beta, ETA_MAX))
        w = mu * theta / (theta + mu)
        grad = X.T @ ((y - mu) * theta / (theta + mu)) - Q @ beta
        H = (X * w[:, None]).T @ X + Q
        step = np.linalg.solve(H, grad)
        for _ in range(30):
            new = objective(beta + step)
            if new >= current - 1e-10:
                break
            step /= 2
        beta = beta + step
        done = abs(new - current) < IRLS_TOL * (abs(current) + 1)
        current = new
        if done:
            break
    mu = np.exp(np.minimum(offset + X @ beta, ETA_MAX))
    w = mu * theta / (theta + mu)
    return beta, (X * w[:, None]).T @ X + Q, current


@dataclass
class Fit:
    design: Design
    hyper: dict
    beta: np.ndarray
    H: np.ndarray
    log_marginal: float

    @property
    def theta(self) -> float:
        return float(np.exp(self.hyper["theta"]))

    def draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Posterior draws of the coefficients, `(n, k)`."""
        L = np.linalg.cholesky(self.H)
        z = rng.standard_normal((len(self.beta), n))
        return self.beta + np.linalg.solve(L.T, z).T


def _laplace(design, X, offset, y, h, beta):
    Q = design.precision(h)
    beta, H, pll = posterior_mode(X, offset, y, Q, np.exp(h["theta"]), beta)
    log_marginal = pll + 0.5 * np.linalg.slogdet(Q)[1] - 0.5 * np.linalg.slogdet(H)[1]
    return log_marginal, beta, H


def fit(variant: str, frame: pd.DataFrame, years: np.ndarray, doy_range, hyper=None) -> Fit:
    """Fit a variant to the observed rows of `frame` (`c > 0`), over the year range `years`.

    With `hyper` given, only the posterior mode is found (no hyperparameter search): how the
    benchmark refits a year with part of it hidden.
    """
    d = design(variant, years, doy_range)
    obs = frame[frame["c"] > 0]
    X = d.X(obs["year"].to_numpy(), obs["doy"].to_numpy())
    offset = np.log(obs["c"].to_numpy())
    y = obs["y"].to_numpy(float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum() / obs["c"].sum(), 1e-3))
    if hyper is None:
        state = {"beta": beta}

        def negative(x):
            h = dict(zip(d.hyper_names, x))
            lm, b, _ = _laplace(d, X, offset, y, h, state["beta"])
            state["beta"] = b
            return -lm

        res = optimize.minimize(
            negative,
            d.hyper_start,
            method="L-BFGS-B",
            bounds=d.hyper_bounds,
            options={"eps": 1e-4, "maxiter": 300},
        )
        hyper, beta = dict(zip(d.hyper_names, res.x)), state["beta"]
    lm, beta, H = _laplace(d, X, offset, y, hyper, beta)
    return Fit(d, dict(hyper), beta, H, lm)


# --- predictions ------------------------------------------------------------


def eta_draws(f: Fit, year, doy, beta_draws, without=()) -> np.ndarray:
    """Log rate per full-day equivalent, `(n_draws, n_rows)`, optionally leaving blocks out."""
    X = f.design.X(np.asarray(year), np.asarray(doy))
    if without:
        X = X.copy()
        X[:, f.design.columns_of(*without)] = 0
    return np.minimum(beta_draws @ X.T, ETA_MAX)


def fill_draws(f: Fit, frame: pd.DataFrame, c_total, n: int = DRAWS, seed: int = 0) -> np.ndarray:
    """Draws of each row's birds over coverage `c_total` (1: the whole day), `(n, n_rows)`.

    The observed `y` (over `c`) is kept and the rest, `c_total - c`, drawn from the posterior
    predictive: the day's rate is gamma(theta, theta / mu) a priori, so given y birds in c it is
    gamma(theta + y, theta / mu + c), and the birds missed are Poisson of that rate times the
    coverage missed. A day with nothing counted is a plain negative binomial draw. `extra` birds
    (counted below `MIN_COVERAGE`) are added as they are.
    """
    rng = np.random.default_rng(seed)
    beta = f.draws(n, rng)
    mu = np.exp(eta_draws(f, frame["year"], frame["doy"], beta))
    y, c = frame["y"].to_numpy(float), frame["c"].to_numpy(float)
    missing = np.clip(np.asarray(c_total, float) - c, 0, None)
    rate = rng.gamma(f.theta + y, 1 / (f.theta / mu + c))
    extra = frame["extra"].to_numpy(float) if "extra" in frame else 0.0
    return y + extra + rng.poisson(rate * missing)


def annual_totals(f: Fit, frame: pd.DataFrame, n: int = DRAWS, seed: int = 0) -> pd.DataFrame:
    """Gap-filled window total per year (observed birds + the posterior predictive of the hours and
    days not counted), its quantiles, the observed share, and the smooth expected total: the trend
    without the year's own level, every hour counted."""
    totals = fill_draws(f, frame, 1.0, n, seed)
    years = frame["year"].to_numpy()
    uy = np.unique(years)
    per_year = np.stack([totals[:, years == y].sum(axis=1) for y in uy], axis=1)
    beta = f.draws(n, np.random.default_rng(seed + 1))
    smooth = np.exp(eta_draws(f, frame["year"], frame["doy"], beta, without=("year",)))
    smooth = np.stack([smooth[:, years == y].sum(axis=1) for y in uy], axis=1)
    counted = frame["y"] + frame.get("extra", 0.0)
    out = pd.DataFrame({"year": uy, "observed": counted.groupby(frame["year"]).sum().to_numpy()})
    out["total"] = per_year.mean(axis=0)
    for q in QUANTILES:
        out[f"q{q * 100:g}"] = np.quantile(per_year, q, axis=0)
    out["observed_share"] = out["observed"] / out["total"]
    out["smooth"] = np.median(smooth, axis=0)
    out["smooth_q2.5"], out["smooth_q97.5"] = np.quantile(smooth, [0.025, 0.975], axis=0)
    return out


def season_curve(f: Fit, year: int, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Expected birds per full day by day of year in `year` (median and 95% band)."""
    doy = np.arange(f.design.doy_range[0], f.design.doy_range[1] + 1)
    beta = f.draws(n, np.random.default_rng(seed))
    mu = np.exp(eta_draws(f, np.full(len(doy), year), doy, beta, without=("year",)))
    q = np.quantile(mu, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({"doy": doy, "lo": q[0], "mid": q[1], "hi": q[2]})


def peak_doy(f: Fit, years: np.ndarray, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Median passage date per year (the day by which half the season's birds have passed)."""
    doy = np.arange(f.design.doy_range[0], f.design.doy_range[1] + 1)
    beta = f.draws(n, np.random.default_rng(seed))
    rows = []
    for yr in years:
        mu = np.exp(eta_draws(f, np.full(len(doy), yr), doy, beta))
        cum = mu.cumsum(axis=1) / mu.sum(axis=1, keepdims=True)
        med = doy[(cum < 0.5).sum(axis=1)]
        rows.append(
            {
                "year": yr,
                "lo": np.quantile(med, 0.1),
                "mid": np.median(med),
                "hi": np.quantile(med, 0.9),
            }
        )
    return pd.DataFrame(rows)
