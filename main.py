"""gta6-trophy-rarity-forecast-01: forecast each GTA VI (PS5) trophy's Exophase
earn rate at the case's target, as five quantiles per trophy.

Program: `python3 /bot/main.py` (Python 3.13, standard library only). It reads
every `<id>.in` case file in the input directory ($MAKEFUN_INPUT_DIR, else
/input) and writes one `<id>.out` forecast per case to the output directory
($MAKEFUN_OUTPUT_DIR, else /output): {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}, ...]}, one row per trophy, quantiles non-decreasing, each in
[0, 100]. No network, no randomness: two runs write identical bytes.

Pipeline (per case, per trophy):
  1. Parse the case robustly (null rates skipped; a trophy with no observed
     rate falls back to the cross-trophy median of last rates, wide spread).
  2. Build the trophy's logit-rate series over the case history, plus the
     players-tracked series.
  3. Forecast a median logit and a logit spread with the selected model
     (MODEL below), then q_tau = sigmoid(median_logit + z_tau * sigma),
     rounded to 4 decimals (half up, as the kit starter does).

MODEL SELECTION -- FINAL (see docs/MODEL-DECISION.md).
The user decided (2026-11-19) that all five model questions are answered
from GTA V data, the only prior available. The shipped family is a
combination: poisson-hazard median, regularized by a GTA V per-type prior
(saturation levels from Exophase PS3 cumulative rarity), with the shared
spread. All three families stay implemented behind one switch for
experiments; the grader runs with no environment, so the in-file MODEL is
what ships:

  MODEL = "poisson-hazard"    # FINAL: decaying-hazard fit on earner counts
                              # + log-linear players-tracked forecast, with
                              # GTA V per-type E_inf cap + weak log-lambda
                              # prior (see LAMBDA_PRIOR / PRIOR_SATURATION).
  MODEL = "logit-linear"      # shrunk OLS of logit(rate) on hours, fitted
                              # on the last 24 h of reads (not shipped).
  MODEL = "prior-shape"       # last rate x a baked per-tier shape factor
                              # (all factors 1.0 = carry-forward; not
                              # shipped -- cumulative GTA V rarity is the
                              # wrong object for a level prior).

Override for local experiments only: MAKEFUN_MODEL=<name>. The grader runs
with no environment, so the in-file MODEL is what ships.

Spread (shared by all three models, so the models differ only in median):
  sigma^2 = (0.5*sqrt(horizon_days))^2            random-walk term (house
                                                   baseline's sigma)
          + stale^2                              staleness term: the page
                                                   can sit unchanged for
                                                   hours while Exophase's
                                                   next refresh jumps every
                                                   rate (launch night: 638
                                                   -> 3,868 players tracked
                                                   at the 04:30Z refresh).
                                                   stale = STALE_SIGMA *
                                                   exp(-gap_hours /
                                                   STALE_TAU), where
                                                   gap_hours is the time
                                                   since that trophy's rate
                                                   last changed (or the whole
                                                   history span if it never
                                                   changed).
          + (se_slope * horizon_hours)^2         trend-uncertainty term
                                                   (logit-linear only).
  sigma is capped at 2.5 logit units. On the flat practice histories every
  model collapses to a carry-forward median (no movement to fit); only the
  spread differs from the house baseline, which is exactly the point: a
  refresh the history has not seen must be inside the forecast's band.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (single-file
# entry, but keep the starter's convention so a helper module could be added).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# ---------------------------------------------------------------------------
# Model selection (FINAL -- see docs/MODEL-DECISION.md).
# ---------------------------------------------------------------------------
MODEL = os.environ.get("MAKEFUN_MODEL", "poisson-hazard")
MODELS = ("logit-linear", "poisson-hazard", "prior-shape")

# --- shared spread parameters (FINAL: kept from provisional) ---------------
SIGMA_RW = 0.5          # logit units per sqrt(day), house baseline's value.
STALE_SIGMA = 1.5       # logit spread added when the page has not changed
                        # at all (a refresh is overdue); decays as fresh
                        # changes arrive.
STALE_TAU_HOURS = 8.0   # staleness decay time.
SIGMA_CAP = 2.5         # hard cap on total logit spread.
SIGMA_FLOOR = 0.05      # never report a (near-)degenerate band.

# --- logit-linear parameters (not shipped; kept for experiments) -----------
FIT_WINDOW_HOURS = 24.0  # fit on the last 24 h of reads (house curve baseline).
SHRINK_K = 4.0           # slope *= n / (n + SHRINK_K); n = fitted points.
MAX_SLOPE = 0.10         # |logit/hour| cap on the fitted slope.
MAX_SHIFT = 3.0          # |median logit - last logit| cap on extrapolation.

# --- poisson-hazard parameters (FINAL: GTA V prior, docs/MODEL-DECISION.md)
HAZARD_TAU_HOURS = 72.0  # hazard decay time h(t) = h0 * exp(-t / TAU).
# Grid of decay rates lam = 1/TAU_eff searched per trophy (per hour).
# The grid MUST reach the saturated regime (lam >> 1 / observed span):
# flat earners are best explained by fast saturation (large lam), and a grid
# truncated below that forces a spurious rising forecast (small lam with a
# huge E_inf extrapolates ~linear growth past the data). Range here covers
# timescales from ~1 minute (lam=100) to ~1 year (lam=1e-4).
HAZARD_LAM_GRID = tuple(10.0 ** (-4.0 + 0.25 * i) for i in range(25))

# --- GTA V prior (FINAL, 2026-11-19; source: Exophase PS3 cumulative rarity,
# fetched 2026-11-19 -- see docs/MODEL-DECISION.md for the table + mapping).
# GTA V is the only prior data: its trackers hold TODAY's cumulative rarity
# (13-year equilibrium), not a launch-week curve, so it is used as a SHAPE
# prior (relative saturation per trophy TYPE) with the LEVEL taken from GTA
# VI's own current rates -- never as a level. The 55 GTA VI trophies are
# grouped by kind (guide data) into six prior groups; each group's saturation
# S is the MEDIAN Exophase PS3 cumulative rarity of its GTA V analogs:
#   story: 11 GTA V story missions, median 56.00%.
#   robbery: 5 GTA V heist story missions, median 57.86%.
#   side: 15 GTA V side activities / economy / exploration / stranger
#     trophies pooled, median 20.38% (activity, stranger, misc, exploration,
#     economy, combat kinds -- no single clean analog each, so pooled).
#   collectible: 3 GTA V collectible sets, median 10.72% (tight: 9.96-10.91).
#   grind: 70 gold medals 5.70%, 100% completion 7.03%, Close Shave 6.64%,
#     median 6.64% (gold-medal activities + 100% completion).
#   platinum: 3.90% (n=1).
# LAMBDA_PRIOR is the weak centre of a log-lambda penalty (see
# model_poisson_hazard): tau = time to 63% of saturation, set by the content's
# nature -- story chapters finish in ~1 day, heists ~2 days, side content
# ~3 days, collectibles ~1 week, grind ~2 weeks, platinum ~30 days. These
# taus are a STATED ASSUMPTION (no launch-week curve exists to estimate them
# from; the Wayback snapshot of the Exophase page at 2013-10-29 exists but
# could not be fetched). The penalty weight is deliberately weak: it only
# breaks ties on flat history and is overruled by real movement.
TROPHY_PRIOR = {
    1: "platinum",
    2: "story", 3: "story", 4: "story", 5: "story", 6: "story", 7: "story",
    8: "story", 9: "story", 10: "story", 11: "story", 12: "story", 13: "story",
    14: "grind", 15: "grind", 17: "grind",
    16: "side", 18: "side", 19: "side", 20: "side", 21: "side", 22: "side",
    23: "side", 24: "side", 25: "side", 26: "side", 27: "side", 28: "side",
    29: "side", 32: "side", 33: "side", 35: "side", 36: "side", 37: "side",
    38: "side", 42: "side", 44: "side", 45: "side", 46: "side", 47: "side",
    48: "side", 49: "side", 50: "side", 51: "side", 52: "side", 53: "side",
    54: "side", 55: "side",
    30: "collectible", 31: "collectible",
    34: "robbery", 39: "robbery", 40: "robbery", 41: "robbery", 43: "robbery",
}
PRIOR_SATURATION = {"story": 0.5600, "robbery": 0.5786, "side": 0.2038,
                    "collectible": 0.1072, "grind": 0.0664, "platinum": 0.0390}
LAMBDA_PRIOR = {"story": 0.0417, "robbery": 0.0208, "side": 0.0139,
                "collectible": 0.0060, "grind": 0.0030, "platinum": 0.0014}
E_INF_CAP_MULT = 1.5   # fitted E_inf may not exceed 1.5x the group's GTA V
                       # saturation (as a rate at target), nor the observed
                       # earner count -- whichever is larger.
LAMBDA_PENALTY_W = 0.001  # weight of the weak log-lambda prior, in relative
                          # SSE units (see model_poisson_hazard).
P_SLOPE_CAP = 0.01        # |log players-tracked/hour| cap on the P(t) fit.
                          # The provisional 0.05/h lets P(t) explode: a
                          # single early refresh jump (638 -> 3,868 in ~5 h)
                          # gives a 48 h OLS slope near 0.04/h, which
                          # extrapolates to ~100x players by day 7 and
                          # collapses every median toward zero. 0.01/h still
                          # allows 3.3x denominator growth over the 5-day
                          # horizon (plausible launch-week surge) while
                          # bounding the dilution. JUDGMENT CALL (no GTA V
                          # P(t) curve exists to calibrate it) -- flagged in
                          # docs/MODEL-DECISION.md.

# --- prior-shape parameters (not shipped: all 1.0 = carry-forward) ---------
# No credible week-one launch curve is baked in (see docs/MODEL-DECISION.md):
# cumulative GTA IV/V rarities are the wrong object (long-run equilibrium,
# not a week-one curve). Until a launch-week curve exists, every factor is
# 1.0 and this model is carry-forward with the shared spread. The GTA V prior
# is used inside the poisson-hazard model instead (PRIOR_SATURATION above).
SHAPE_FACTOR = {"platinum": 1.0, "gold": 1.0, "silver": 1.0, "bronze": 1.0}

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation, |err| < 1.2e-9)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, se_b, n). se_b = 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.0, n)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        dof = n - 2
        se_b = math.sqrt(rss / dof / sxx) if rss > 0 else 0.0
    else:
        se_b = 0.0
    return (a, b, se_b, n)


# ---------------------------------------------------------------------------
# Case parsing
# ---------------------------------------------------------------------------
class TrophySeries:
    """One trophy's observed history: (hours, logit rate) points, last rate,
    and the hours since its rate last changed (staleness)."""

    def __init__(self, idx, case, t0):
        self.idx = idx
        self.points = []          # (hours since t0, logit rate)
        self.last_rate = None
        self.last_change_gap_h = None
        last_seen = None
        last_change_t = None
        for h in case["history"]:
            rates = h.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            rate = float(rates[idx])
            t = (parse_time(h["at"]) - t0).total_seconds() / 3600.0
            self.points.append((t, logit(rate)))
            self.last_rate = rate
            if last_seen is None or rate != last_seen:
                last_change_t = t
            last_seen = rate
        if self.points:
            t_end = (parse_time(case["at"]) - t0).total_seconds() / 3600.0
            t_start = self.points[0][0]
            if last_change_t is None:
                self.last_change_gap_h = t_end - t_start
            else:
                self.last_change_gap_h = max(0.0, t_end - last_change_t)


def spread_sigma(case, series, horizon_h, se_slope=0.0):
    """Shared logit spread: random walk + staleness + trend uncertainty."""
    days = horizon_h / 24.0
    rw = SIGMA_RW * math.sqrt(days)
    gap = series.last_change_gap_h
    stale = STALE_SIGMA * math.exp(-gap / STALE_TAU_HOURS) if gap is not None else STALE_SIGMA
    trend = abs(se_slope) * horizon_h
    sigma = math.sqrt(rw * rw + stale * stale + trend * trend)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


# ---------------------------------------------------------------------------
# The three candidate model families. Each returns (median_logit, sigma).
# ---------------------------------------------------------------------------
def model_logit_linear(case, series, horizon_h, t0):
    """Not shipped (see MODEL): shrunk OLS of logit(rate) on hours over the last
    FIT_WINDOW_HOURS of reads, extrapolated to the target. Flat history ->
    slope 0 -> carry-forward median."""
    t_end = (parse_time(case["at"]) - t0).total_seconds() / 3600.0
    pts = [(t, y) for (t, y) in series.points if t >= t_end - FIT_WINDOW_HOURS]
    if len(pts) < 3:
        pts = series.points
    if len(pts) >= 2 and series.last_rate is not None:
        xs = [t for (t, _) in pts]
        ys = [y for (_, y) in pts]
        _, slope, se_b, n = ols(xs, ys)
        slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope)) * n / (n + SHRINK_K)
        last_lg = logit(series.last_rate)
        median_lg = last_lg + slope * horizon_h
        median_lg = max(last_lg - MAX_SHIFT, min(last_lg + MAX_SHIFT, median_lg))
        return median_lg, spread_sigma(case, series, horizon_h, se_slope=se_b)
    if series.last_rate is not None:
        return logit(series.last_rate), spread_sigma(case, series, horizon_h)
    return logit(1.0), SIGMA_CAP  # no observations at all: wide, centred low


def model_poisson_hazard(case, series, horizon_h, t0):
    """Decaying-hazard fit on earner COUNTS, regularized by a GTA V prior
    (FINAL -- see docs/MODEL-DECISION.md). E(t) = E_inf * (1 - exp(-lam*t)),
    t hours since release; lam picked by penalized least squares over a grid,
    E_inf by linear scale at that lam. P(t) is forecast by log-linear fit on
    players_tracked. Rate = 100 * E / P -- so the median shifts DOWN when the
    tracked-player denominator grows (the refresh mechanism, modelled
    explicitly, not an ad-hoc offset). The GTA V prior enters twice:
    (1) a hard cap E_inf <= max(E_last, E_INF_CAP_MULT * S_group * P_target),
    where S_group is the GTA V median long-run saturation of the trophy's
    type (Exophase PS3) -- the cap bounds extrapolated growth, never the
    observed level; (2) a weak log-lambda penalty pulling lam toward the
    group's prior centre, which only breaks ties on flat history and is
    overruled by real movement. Flat history -> saturated fit, E_inf ~ E_last,
    P ~ P_last: carry-forward."""
    release = parse_time(case.get("game", {}).get("release_date", case["history"][0]["at"]))
    hist = case["history"]
    if series.last_rate is None or not hist:
        return logit(1.0), SIGMA_CAP
    # GTA V prior group for this trophy (by id; default "side").
    tid = case["trophies"][series.idx].get("id", 0)
    group = TROPHY_PRIOR.get(tid, "side")
    sat = PRIOR_SATURATION[group]
    lam0 = LAMBDA_PRIOR[group]
    # Earner counts per read.
    obs = []
    for h in hist:
        rates = h.get("rates") or []
        if series.idx >= len(rates) or rates[series.idx] is None:
            continue
        players = h.get("players_tracked")
        if not players:
            continue
        t = (parse_time(h["at"]) - release).total_seconds() / 3600.0
        if t <= 0:
            continue
        obs.append((t, float(rates[series.idx]) / 100.0 * float(players)))
    if len(obs) < 2:
        return logit(series.last_rate), spread_sigma(case, series, horizon_h)
    # Players-tracked forecast first: the E_inf cap is a rate cap at target.
    t_target = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    p_pts = [((parse_time(h["at"]) - release).total_seconds() / 3600.0,
              math.log(float(h["players_tracked"])))
             for h in hist if h.get("players_tracked")]
    p_target = float(hist[-1]["players_tracked"])
    if len(p_pts) >= 2:
        _, b, _, _ = ols([t for (t, _) in p_pts], [y for (_, y) in p_pts])
        b = max(-P_SLOPE_CAP, min(P_SLOPE_CAP, b))
        t_last = p_pts[-1][0]
        p_target = math.exp(math.log(p_target) + b * max(0.0, t_target - t_last))
        p_target = max(p_target, 1.0)
    e_inf_cap = max(obs[-1][1], E_INF_CAP_MULT * sat * p_target)
    mean_e = sum(e for (_, e) in obs) / len(obs)
    scale = len(obs) * mean_e * mean_e if mean_e > 0 else 0.0
    best = None
    for lam in HAZARD_LAM_GRID:
        f = [1.0 - math.exp(-lam * t) for (t, _) in obs]
        denom = sum(v * v for v in f)
        if denom <= 0:
            continue
        e_inf = sum(e * v for (_, e), v in zip(obs, f)) / denom
        if e_inf < 0 or e_inf > e_inf_cap:
            continue
        sse = sum((e - e_inf * v) ** 2 for (_, e), v in zip(obs, f))
        # Penalized relative score: plain least squares, plus a weak prior
        # pulling log10(lam) toward the group's centre. The penalty is
        # dimensionless (SSE scaled by n * mean_e^2); it breaks ties on flat
        # history and is overruled whenever the data prefers another lam by
        # more than the penalty weight.
        if scale > 0:
            rel = sse / scale + LAMBDA_PENALTY_W * (math.log10(lam) - math.log10(lam0)) ** 2
        else:
            rel = sse
        if best is None or rel < best[0]:
            best = (rel, lam, e_inf)
    if best is None:
        # Every grid point violated the cap: fall back to carry-forward.
        return logit(series.last_rate), spread_sigma(case, series, horizon_h)
    _, lam, e_inf = best
    e_target = e_inf * (1.0 - math.exp(-lam * max(t_target, 0.0)))
    rate = clamp(100.0 * max(e_target, 0.0) / p_target)
    return logit(rate), spread_sigma(case, series, horizon_h)


def model_prior_shape(case, series, horizon_h, t0):
    """Baked shape prior: last rate x per-tier factor. All factors are 1.0
    (carry-forward) because no credible week-one launch curve exists to bake
    in -- cumulative GTA IV/V rarities are long-run equilibria, the wrong
    object for a 4.5-day forecast. See docs/MODEL-DECISION.md."""
    tier = case["trophies"][series.idx].get("tier", "bronze")
    factor = SHAPE_FACTOR.get(tier, 1.0)
    if series.last_rate is not None:
        return logit(clamp(series.last_rate * factor)), spread_sigma(case, series, horizon_h)
    return logit(1.0), SIGMA_CAP


MODELS_FN = {
    "logit-linear": model_logit_linear,
    "poisson-hazard": model_poisson_hazard,
    "prior-shape": model_prior_shape,
}


# ---------------------------------------------------------------------------
# Forecast one case
# ---------------------------------------------------------------------------
def forecast_case(case, name):
    if MODEL not in MODELS_FN:
        raise ValueError(f"{name}: unknown MODEL {MODEL!r}, expected one of {MODELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    levels = case.get("levels", LEVELS)
    zs = [norm_ppf(lv) for lv in levels]
    horizon_h = float(case["horizon_minutes"]) / 60.0
    t0 = parse_time(case["history"][0]["at"])
    fn = MODELS_FN[MODEL]
    # Fallback median for trophies never observed: cross-trophy median of
    # last rates (robust, deterministic).
    last_rates = []
    series_list = []
    for i in range(len(case["trophies"])):
        s = TrophySeries(i, case, t0)
        series_list.append(s)
        if s.last_rate is not None:
            last_rates.append(s.last_rate)
    last_rates.sort()
    fallback_rate = last_rates[len(last_rates) // 2] if last_rates else 1.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        s = series_list[i]
        if s.last_rate is None:
            # Never observed: centre on the cross-trophy median, widest band.
            median_lg, sigma = logit(fallback_rate), SIGMA_CAP
        else:
            median_lg, sigma = fn(case, s, horizon_h, t0)
        qs = []
        prev = None
        for z in zs:
            q = round4(clamp(sigmoid(median_lg + z * sigma), 0.0, 100.0))
            # Keep quantiles non-decreasing after rounding.
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit(f"no *.in case files in {INPUT_DIR}")
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
