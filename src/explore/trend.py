"""Trend model for the Explore page: one taxon's smooth trend, season and phenology shift, and
gap-filled annual totals. Part of `src.explore` (see its `__init__` for the boundary).

A GAM, benchmarked by `scripts/benchmark_trend.py` (`DECISIONS.md` -> Explore has why a GAM):

    count_day ~ NegBin(mean = c_day * exp(eta), theta)
    eta = a + trend(year) + year_effect(year) + season(doy) + shift(year, doy)
          + episode_year(doy)

`c_day` is the day's coverage (`src.explore.profile.coverage`: the share of the day's expected
passage in the hours counted), `year_effect` an independent level per year (good and bad years),
`shift` a smooth change of the season's timing or width over the years, and `episode` a short-range
curve of its own each year: runs of good or bad migration days, the weather's share, which a
weather model would one day explain. Without it the shift absorbed them (lengthscales of 3 years by
6 days), and filled days were treated as independent of their neighbours. The negative binomial is a
day's gamma-distributed rate seen through a Poisson count; within the day, passage comes in flocks,
so the hours not counted are drawn with an hourly over-dispersion `kappa` (`hour_dispersion`,
`fill_draws`).

Every component is a basis times coefficients with a Gaussian prior (a penalty): P-splines, cubic
B-spline bases with second-difference penalties (a tensor product for `shift`, first differences
plus a ridge for `episode`), a smoothing parameter per penalty, as mgcv would. The posterior mode is
found by penalised IRLS, and the smoothing parameters and `theta` by the Laplace approximation of
the marginal likelihood (`fit`). Variants:

- `gam`: the model above.
- `season`: the season alone, a fixed phenology without trend or year effect, as the forecast's
  baseline has: the reference the GAM must beat.

Beyond the data (a year not yet counted), the trend extrapolates its last slope: Explore never asks
for it, and a forecast that did would need a flat extrapolation instead.

Uncertainty is the posterior given the fitted hyperparameters (empirical Bayes); intervals for annual
totals are calibrated by the benchmark's gap-transplant test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

from src.explore.export import COMBINED_RANK, HOURS, in_window
from src.explore.profile import PROFILE_DOY, PROFILE_MIN_TIMED, coverage

VARIANTS = ("season", "gam", "weather")
# Ranks whose trend is exported: unidentified birds ("spuh", e.g. falcon sp.) trend with how hard
# observers try to identify them, not with the birds.
TREND_RANKS = ("species", COMBINED_RANK)

# Prior precision on coefficients no penalty reaches (intercept, spline null spaces): effectively
# flat on the log scale (sd 100), and what keeps the overlapping constants identifiable.
RIDGE = 1e-4
ETA_MAX = 20.0  # log birds per full-day equivalent; guards exp() in early IRLS steps

# P-spline bases (cubic, second-order difference penalties).
GAM_TREND_K = 12
GAM_SEASON_K = 20
GAM_SHIFT_K = (6, 8)  # year x doy
GAM_EPISODE_K = 25  # per year, over the window's ~124 days: a knot every ~5 days
GAM_LOG_LAMBDA_BOUNDS = (-6.0, 16.0)
LOG_LAMBDA_START = 2.0
EFS_MAX_ITER = 100
THETA_ROUNDS = 5  # alternations of theta (marginal likelihood) and the smoothing parameters
EFS_TOL = 1e-2  # largest change of a log smoothing parameter or log theta: 1%

# `weather`: the GAM plus a smooth of each of these daily ERA5 values at Défilé (daytime means,
# precipitation a daytime sum on a log scale), from the forecast's local cache.
WEATHER_LOCATION = "Defile"
WEATHER_VARIABLES = (
    "u_component_of_wind_100m",
    "v_component_of_wind_100m",
    "total_precipitation",
    "total_cloud_cover",
    "temperature_2m",
)
WEATHER_HOURS_UTC = (6, 16)  # [start, end): about 8-18 local in the season
WEATHER_K = 8
THETA_BOUNDS = (0.02, 200.0)  # negative binomial shape; small = over-dispersed
KAPPA_BOUNDS = (0.01, 1e4)
# Hours summed into blocks before `kappa` is fitted: a day's passage shifts as a whole (later on a
# slow morning), so neighbouring hours are not independent and `kappa` falls from single hours to
# blocks of 3-4 hours, then levels off (Honey Buzzard 4.7 -> 2.6). Gaps are blocks of hours too.
KAPPA_BLOCK = 4  # hourly over-dispersion, per full day of coverage; large = Poisson
SMOOTH = (
    "year",
    "episode",
    "weather",
)  # left out of the smooth trend, season curves and passage dates

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


def daily_weather(cache_dir: str, years: np.ndarray, doy_range: tuple[int, int]) -> np.ndarray:
    """`WEATHER_VARIABLES` at `WEATHER_LOCATION`, `(n_years, n_doy, n_variables)`, standardised:

    daytime (`WEATHER_HOURS_UTC`) means, precipitation as log1p of the daytime sum in mm.
    """
    from src.data.weather import load_cache

    ds = load_cache(
        cache_dir, [WEATHER_LOCATION], list(WEATHER_VARIABLES), years=years, doy=doy_range
    ).isel(location=0)
    h = (ds["time"].values / np.timedelta64(1, "h")).astype(int)
    day = ds.isel(time=np.flatnonzero((h >= WEATHER_HOURS_UTC[0]) & (h < WEATHER_HOURS_UTC[1])))
    dates = pd.DatetimeIndex(ds["date"].values)
    y0, d0 = int(years.min()), doy_range[0]
    out = np.full((len(years), doy_range[1] - d0 + 1, len(WEATHER_VARIABLES)), np.nan)
    yi, di = dates.year - y0, dates.dayofyear - d0
    ok = (yi >= 0) & (yi < len(years)) & (di >= 0) & (di < out.shape[1])
    for j, v in enumerate(WEATHER_VARIABLES):
        x = day[v].sum("time") * 1000 if v == "total_precipitation" else day[v].mean("time")
        x = np.log1p(np.asarray(x, float)) if v == "total_precipitation" else np.asarray(x, float)
        out[yi[ok], di[ok], j] = x[ok]
    mean, sd = np.nanmean(out, axis=(0, 1)), np.nanstd(out, axis=(0, 1))
    return (out - mean) / sd


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


@dataclass
class Block:
    """A model component: its columns in the design and its penalties, each a smoothing parameter's
    name and the matrix it multiplies (the prior precision is their sum)."""

    name: str
    columns: slice
    penalties: list[tuple[str, np.ndarray]]


@dataclass
class Design:
    variant: str
    years: np.ndarray  # every year of the model's range, observed or not
    doy_range: tuple[int, int]
    blocks: list[Block]
    _basis: callable = field(repr=False)

    def X(self, year: np.ndarray, doy: np.ndarray) -> sparse.csr_matrix:
        return self._basis(np.asarray(year), np.asarray(doy))

    @property
    def n_columns(self) -> int:
        return self.blocks[-1].columns.stop

    @property
    def hyper_names(self) -> list[str]:
        return [name for b in self.blocks for name, _ in b.penalties]

    def block_precision(self, b: Block, h: dict) -> np.ndarray:
        k = b.columns.stop - b.columns.start
        return sum((np.exp(h[n]) * S for n, S in b.penalties), RIDGE * np.eye(k))

    def precision(self, h: dict) -> np.ndarray:
        n = self.n_columns
        Q = np.zeros((n, n))
        for b in self.blocks:
            Q[b.columns, b.columns] = self.block_precision(b, h)
        return Q

    def columns_of(self, *names: str) -> np.ndarray:
        """Indices of the named blocks' columns."""
        return np.concatenate(
            [np.arange(b.columns.start, b.columns.stop) for b in self.blocks if b.name in names]
            + [np.zeros(0, int)]
        )


def design(
    variant: str, years: np.ndarray, doy_range: tuple[int, int], weather: np.ndarray | None = None
) -> Design:
    """The bases and penalties of a variant over `years` (all of them, including years without
    data, whose coefficients then stay at their prior) and the window's days of year.

    Every smooth sums to zero over its grid (`sum_to_zero`), so each level has one owner: the
    intercept, then `year` for a year's level (`episode` has zero mean over each season), and the
    trend and the season for the main effects of `shift` (an interaction only, as mgcv's `ti`).
    """
    y0, y1 = int(years.min()), int(years.max())
    d0, d1 = doy_range
    n_years = y1 - y0 + 1
    year_grid, doy_grid = np.arange(y0, y1 + 1), np.arange(d0, d1 + 1)
    parts, blocks = [], []
    col = 0

    def add(name, basis, penalties):
        nonlocal col
        k = basis(np.array([y0]), np.array([d0])).shape[1]
        parts.append(basis)
        blocks.append(Block(name, slice(col, col + k), penalties))
        col += k

    def centred(lo, hi, k, grid, order=2):
        """A B-spline basis that sums to zero over `grid`, and its difference penalty."""
        Z = sum_to_zero(bspline_basis(grid, lo, hi, k))
        return (lambda v: bspline_basis(v, lo, hi, k) @ Z), Z.T @ difference_penalty(k, order) @ Z

    add("intercept", lambda y, d: np.ones((len(y), 1)), [])
    season, S = centred(d0, d1, GAM_SEASON_K, doy_grid)
    add("season", lambda y, d: season(d), [("season", S)])
    if variant in ("gam", "weather"):
        trend, St = centred(y0, y1, GAM_TREND_K, year_grid)
        add("trend", lambda y, d: trend(y), [("trend", St)])
        ky, kd = GAM_SHIFT_K
        shift_y, Sy = centred(y0, y1, ky, year_grid)
        shift_d, Sd = centred(d0, d1, kd, doy_grid)
        add(
            "shift",
            lambda y, d: _row_kron(shift_y(y), shift_d(d)),
            [
                ("shift_year", np.kron(Sy, np.eye(kd - 1))),
                ("shift_doy", np.kron(np.eye(ky - 1), Sd)),
            ],
        )
        add(
            "year",
            lambda y, d: np.eye(n_years)[np.clip(y - y0, 0, n_years - 1)],
            [("year", np.eye(n_years))],
        )
    if variant == "weather":
        if weather is None:
            raise ValueError("the weather variant needs `weather` (daily_weather)")
        W = np.nan_to_num(weather)  # standardised: missing days at the mean
        bases, penalties, k = [], [], WEATHER_K - 1
        for j, name in enumerate(WEATHER_VARIABLES):
            lo, hi = float(W[..., j].min()), float(W[..., j].max())
            b, S = centred(lo, hi, WEATHER_K, W[..., j].ravel())
            bases.append(b)
            P = np.zeros((len(WEATHER_VARIABLES) * k,) * 2)
            P[j * k : (j + 1) * k, j * k : (j + 1) * k] = S
            penalties.append((f"weather_{j}", P))

        def weather_basis(y, d):
            w = W[np.clip(y - y0, 0, n_years - 1), np.clip(d - d0, 0, W.shape[1] - 1)]
            return np.hstack([b(w[:, j]) for j, b in enumerate(bases)])

        add("weather", weather_basis, penalties)
    if variant in ("gam", "weather"):
        episode, Se = centred(d0, d1, GAM_EPISODE_K, doy_grid, order=1)
        add(
            "episode",
            lambda y, d: _per_year(y - y0, n_years, episode(d)),
            [
                ("episode", np.kron(np.eye(n_years), Se)),
                ("episode_size", np.eye(n_years * (GAM_EPISODE_K - 1))),
            ],
        )

    def basis(y, d):
        return sparse.hstack([sparse.csr_matrix(p(y, d)) for p in parts], format="csr")

    return Design(variant, year_grid, doy_range, blocks, basis)


def sum_to_zero(B: np.ndarray) -> np.ndarray:
    """`Z` (k, k - 1) such that `B @ Z @ b` sums to zero over the rows of `B` for any `b`: the
    centring constraint that separates a smooth from the intercept and from other smooths."""
    q, _ = np.linalg.qr(B.sum(axis=0)[:, None], mode="complete")
    return q[:, 1:]


def _per_year(index: np.ndarray, n_years: int, b: np.ndarray) -> sparse.csr_matrix:
    """`b`'s row placed in the columns of its year: a separate curve per year."""
    n, k = b.shape
    index = np.clip(index, 0, n_years - 1)
    cols = (index[:, None] * k + np.arange(k)).ravel()
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((b.ravel(), (rows, cols)), shape=(n, n_years * k))


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


class Matrix:
    """A design matrix kept as its dense columns and its sparse per-year `episode` columns (the
    last block), so that X'WX, a product over ~4 000 rows by ~1 200 columns, costs the dense part
    plus the episode's block diagonal."""

    def __init__(self, X: sparse.csr_matrix, split: int):
        self.dense = X[:, :split].toarray()
        self.sparse = X[:, split:].tocsr()
        self.split = split

    def __matmul__(self, b: np.ndarray) -> np.ndarray:
        return self.dense @ b[: self.split] + self.sparse @ b[self.split :]

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        return np.concatenate([self.dense.T @ r, self.sparse.T @ r])

    def information(self, w: np.ndarray) -> np.ndarray:
        dw = self.dense * w[:, None]
        cross = np.asarray((self.sparse.T @ dw).T)
        corner = (self.sparse.T @ self.sparse.multiply(w[:, None])).toarray()
        return np.block([[self.dense.T @ dw, cross], [cross.T, corner]])


def posterior_mode(X: Matrix, offset, y, Q, theta, beta):
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
        grad = X.rmatvec((y - mu) * theta / (theta + mu)) - Q @ beta
        H = X.information(w) + Q
        step = cho_solve(cho_factor(H), grad)
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
    return beta, X.information(w) + Q, current


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


def _logdet(A: np.ndarray) -> float:
    return 2 * np.log(np.diag(cho_factor(A, lower=True)[0])).sum()


def _laplace(design, X, offset, y, h, beta):
    Q = design.precision(h)
    beta, H, pll = posterior_mode(X, offset, y, Q, np.exp(h["theta"]), beta)
    log_marginal = pll + 0.5 * _logdet(Q) - 0.5 * _logdet(H)
    return log_marginal, beta, H


def _efs(d: Design, X, offset, y, hyper: dict, beta, theta_ml: bool):
    """Fellner-Schall iterations of the smoothing parameters to convergence, `theta` fixed or (with
    `theta_ml`) its maximum likelihood given the fitted means after each update."""
    for _ in range(EFS_MAX_ITER):
        beta, H, _ = posterior_mode(X, offset, y, d.precision(hyper), np.exp(hyper["theta"]), beta)
        H_inv = cho_solve(cho_factor(H), np.eye(len(beta)))
        new = dict(hyper)
        for b in d.blocks:
            if not b.penalties:
                continue
            Q_inv = np.linalg.inv(d.block_precision(b, hyper))
            Hb, bb = H_inv[b.columns, b.columns], beta[b.columns]
            for name, S in b.penalties:
                room = np.sum(Q_inv * S) - np.sum(Hb * S)  # tr(Q^-1 S) - tr(H^-1 S)
                wiggle = bb @ S @ bb
                step = np.log(max(room, 1e-12) / max(wiggle, 1e-12))
                new[name] = float(np.clip(hyper[name] + step, *GAM_LOG_LAMBDA_BOUNDS))
        if theta_ml:
            mu = np.exp(np.minimum(offset + X @ beta, ETA_MAX))
            new["theta"] = float(
                optimize.minimize_scalar(
                    lambda t: -nb_loglik(y, mu, np.exp(t)).sum(),
                    bounds=np.log(THETA_BOUNDS),
                    method="bounded",
                ).x
            )
        change = max(abs(new[k] - hyper[k]) for k in hyper)
        hyper = new
        if change < EFS_TOL:
            break
    return hyper, beta


def fit(
    variant: str,
    frame: pd.DataFrame,
    years: np.ndarray,
    doy_range,
    hyper=None,
    weather: np.ndarray | None = None,
) -> Fit:
    """Fit a variant to the observed rows of `frame` (`c > 0`), over the year range `years`.

    The smoothing parameters by the extended Fellner-Schall update (Wood and Fasiolo 2017; mgcv's
    `optimizer = "efs"`), a fixed point of the Laplace marginal likelihood: each penalty's
    `lambda` is scaled by `(tr(Q^-1 S) - tr(H^-1 S)) / (beta' S beta)`, the degrees of freedom its
    prior allows minus those the data use, over the wiggliness it is asked to explain. `theta`
    starts at its maximum likelihood given the fitted means, which over-fit means make too large
    (days look less variable than they are), then alternates with the smoothing parameters as the
    maximum of the Laplace marginal likelihood. With `hyper` given, only the posterior mode is
    found: how the benchmark refits a year with part of it hidden.
    """
    d = design(variant, years, doy_range, weather)
    obs = frame[frame["c"] > 0]
    episode = [b.columns.start for b in d.blocks if b.name == "episode"]
    X = Matrix(d.X(obs["year"].to_numpy(), obs["doy"].to_numpy()), (episode or [d.n_columns])[0])
    offset = np.log(obs["c"].to_numpy())
    y = obs["y"].to_numpy(float)
    beta = np.zeros(d.n_columns)
    beta[0] = np.log(max(y.sum() / obs["c"].sum(), 1e-3))
    if hyper is None:
        hyper = {name: LOG_LAMBDA_START for name in d.hyper_names} | {"theta": 0.0}
        hyper, beta = _efs(d, X, offset, y, hyper, beta, theta_ml=True)
        for _ in range(THETA_ROUNDS):
            state = {"beta": beta}

            def negative(t):
                lm, b, _ = _laplace(d, X, offset, y, hyper | {"theta": t}, state["beta"])
                state["beta"] = b
                return -lm

            t = hyper["theta"]
            lo, hi = np.log(THETA_BOUNDS)
            theta = optimize.minimize_scalar(
                negative,
                bounds=(max(lo, t - 2), min(hi, t + 1)),
                method="bounded",
                options={"xatol": EFS_TOL},
            ).x
            hyper, beta = _efs(d, X, offset, y, hyper | {"theta": float(theta)}, beta, False)
            if abs(theta - t) < EFS_TOL:
                break
    lm, beta, H = _laplace(d, X, offset, y, hyper, beta)
    return Fit(d, dict(hyper), beta, H, lm)


# --- predictions ------------------------------------------------------------


def eta_draws(f: Fit, year, doy, beta_draws, without=()) -> np.ndarray:
    """Log rate per full-day equivalent, `(n_draws, n_rows)`, optionally leaving blocks out."""
    X = f.design.X(np.asarray(year), np.asarray(doy)).tocsc()
    if without:
        keep = np.ones(X.shape[1])
        keep[f.design.columns_of(*without)] = 0
        X = X @ sparse.diags(keep)
    return np.minimum(np.asarray((X @ beta_draws.T).T), ETA_MAX)


def hour_dispersion(
    days: pd.DataFrame,
    hourly: pd.DataFrame,
    effort: pd.DataFrame,
    profile: np.ndarray,
    block: int = KAPPA_BLOCK,
) -> float:
    """Hourly over-dispersion `kappa` of one taxon: within a day, the passage of a coverage `m` is.

    gamma(kappa * m, kappa / rate) a priori, so given the day's birds, the birds of its counted
    hours are Dirichlet-multinomial with weights kappa * p(h) * hours counted. Maximum likelihood
    over the days timed to the hour (`PROFILE_MIN_TIMED`) with at least two counted hours; birds
    timed to an hour with no coverage are left out. Without such days: `KAPPA_BOUNDS[1]`, Poisson.
    Fitted on blocks of `block` clock hours (`KAPPA_BLOCK`), at least two of them counted.
    """
    d = days[(days["count"] > 0) & (days["timed"] >= PROFILE_MIN_TIMED)]
    counted = effort[effort["state"] == "counted"].set_index("date")["hourly"]
    d = d[d["date"].isin(counted.index)].reset_index(drop=True)
    if d.empty:
        return KAPPA_BOUNDS[1]
    pos = pd.Series(d.index, index=d["date"])
    h = hourly[hourly["date"].isin(pos.index)]
    birds = np.zeros((len(d), HOURS))
    np.add.at(birds, (pos[h["date"]].to_numpy(), h["hour"].to_numpy()), h["count"].to_numpy())
    doy = d["date"].dt.dayofyear.clip(*PROFILE_DOY) - PROFILE_DOY[0]
    a = profile[doy.to_numpy()] * np.stack(d["date"].map(counted).to_numpy())
    on = a > 0
    birds = np.where(on, birds, 0)
    if block > 1:  # clock hours summed into blocks: a Dirichlet's kappa is unchanged by this
        a = a.reshape(len(a), -1, block).sum(axis=2)
        birds = birds.reshape(len(birds), -1, block).sum(axis=2)
        on = a > 0
    ok = (on.sum(axis=1) >= 2) & (birds.sum(axis=1) > 0)
    a, birds, on = a[ok], birds[ok], on[ok]
    if not len(a):
        return KAPPA_BOUNDS[1]
    total = birds.sum(axis=1)

    def nll(log_k):
        alpha = np.exp(log_k) * a
        A = alpha.sum(axis=1)
        per_hour = np.where(on, gammaln(birds + alpha) - gammaln(np.where(on, alpha, 1)), 0)
        return -(gammaln(A) - gammaln(total + A) + per_hour.sum(axis=1)).sum()

    res = optimize.minimize_scalar(nll, bounds=np.log(KAPPA_BOUNDS), method="bounded")
    return float(np.exp(res.x))


def fill_draws(
    f: Fit, frame: pd.DataFrame, c_total, n: int = DRAWS, seed: int = 0, kappa: float | None = None
) -> np.ndarray:
    """Draws of each row's birds over coverage `c_total` (1: the whole day), `(n, n_rows)`.

    The observed `y` (over `c`) is kept and the rest, `c_total - c`, drawn from the posterior
    predictive: the day's rate is gamma(theta, theta / mu) a priori, so given y birds in c it is
    gamma(theta + y, theta / mu + c), and the birds missed are Poisson of that rate times the
    coverage missed, or with `kappa` (`hour_dispersion`) of a gamma(kappa * missed, kappa / rate)
    passage: flocks. A day with nothing counted is a plain negative binomial draw (with `kappa`, a
    little wider). `extra` birds (counted below `MIN_COVERAGE`) are added as they are.

    The day's rate given y treats the counted hours as Poisson. Treating them as flocks too (y
    negative binomial of size kappa * c, which shrinks y and c by kappa c / (kappa c + y)) was
    tried: it biased gap-filled totals 3% low (`DECISIONS.md` -> Explore).
    """
    rng = np.random.default_rng(seed)
    beta = f.draws(n, rng)
    mu = np.exp(eta_draws(f, frame["year"], frame["doy"], beta))
    y, c = frame["y"].to_numpy(float), frame["c"].to_numpy(float)
    missing = np.clip(np.asarray(c_total, float) - c, 0, None)
    rate = rng.gamma(f.theta + y, 1 / (f.theta / mu + c))
    passage = rate * missing
    if kappa is not None:
        passage = rng.gamma(kappa * missing + 1e-300, passage / (kappa * missing + 1e-300))
    extra = frame["extra"].to_numpy(float) if "extra" in frame else 0.0
    return y + extra + rng.poisson(passage)


def annual_totals(
    f: Fit, frame: pd.DataFrame, n: int = DRAWS, seed: int = 0, kappa: float | None = None
) -> pd.DataFrame:
    """Gap-filled window total per year (observed birds + the posterior predictive of the hours and
    days not counted), its quantiles, the observed share, and the smooth expected total: the trend
    without the year's own level, every hour counted."""
    totals = fill_draws(f, frame, 1.0, n, seed, kappa)
    years = frame["year"].to_numpy()
    uy = np.unique(years)
    per_year = np.stack([totals[:, years == y].sum(axis=1) for y in uy], axis=1)
    beta = f.draws(n, np.random.default_rng(seed + 1))
    smooth = np.exp(eta_draws(f, frame["year"], frame["doy"], beta, without=SMOOTH))
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
    mu = np.exp(eta_draws(f, np.full(len(doy), year), doy, beta, without=SMOOTH))
    q = np.quantile(mu, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({"doy": doy, "lo": q[0], "mid": q[1], "hi": q[2]})


def peak_doy(f: Fit, years: np.ndarray, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Median passage date per year (the day by which half the season's birds have passed), of
    the smooth season: without the year's weather episodes."""
    doy = np.arange(f.design.doy_range[0], f.design.doy_range[1] + 1)
    beta = f.draws(n, np.random.default_rng(seed))
    rows = []
    for yr in years:
        mu = np.exp(eta_draws(f, np.full(len(doy), yr), doy, beta, without=SMOOTH))
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


# --- export -----------------------------------------------------------------


def taxon_trend(
    days: pd.DataFrame,
    hourly: pd.DataFrame,
    effort: pd.DataFrame,
    profile: np.ndarray,
    first_year: int,
    last_year: int,
    variant: str = "gam",
    seed: int = 0,
) -> dict:
    """One taxon's trend for `species/<taxon_id>.json`, `first_year`..`last_year` (complete seasons
    from its start year), in the default window.

    `annual`: per year, birds counted, the gap-filled total (median and 80%/95% intervals), the
    share counted, and the smooth expected total (trend without the year's level and weather
    episodes, every hour counted) with its 95% band. `passage`: the smooth season's median passage
    date (day of year, 80% band). `season`: expected birds per full day on each day of year,
    smooth, in the first and the last year.
    """
    frame = model_frame(days, effort, coverage(effort, profile), first_year, last_year)
    years = np.arange(first_year, last_year + 1)
    doy_range = (int(frame["doy"].min()), int(frame["doy"].max()))
    kappa = hour_dispersion(days, hourly, effort, profile)
    f = fit(variant, frame, years, doy_range)
    a = annual_totals(f, frame, seed=seed, kappa=kappa)
    a = a.drop(columns=["total"]).rename(columns={"q50": "total"})
    a["observed_share"] = a["observed"] / a["total"]
    birds = [c for c in a.columns if c not in ("year", "observed_share")]
    a[birds] = a[birds].round(0)
    curves = {
        str(y): season_curve(f, y, seed=seed)["mid"].to_numpy() for y in (first_year, last_year)
    }
    return {
        "model": variant,
        "first_year": first_year,
        "last_year": last_year,
        "theta": f.theta,
        "kappa": kappa,
        "annual": a,
        "passage": peak_doy(f, years, seed=seed),
        "season": {"doy": np.arange(doy_range[0], doy_range[1] + 1), **curves},
    }
