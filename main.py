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

  MODEL = "poisson-hazard"    # FINAL v2: decaying-hazard fit on earner
                              # counts + denominator as a SCENARIO MIXTURE
                              # (refresh probability + size), with GTA V
                              # per-type E_inf cap + weak log-lambda prior
                              # (see LAMBDA_PRIOR / PRIOR_SATURATION).
                              # v1's log-linear P(t) + P_SLOPE_CAP is deleted
                              # (see docs/MODEL-DECISION.md).
  MODEL = "logit-linear"      # shrunk OLS of logit(rate) on hours, fitted
                              # on the last 24 h of reads (not shipped).
  MODEL = "prior-shape"       # last rate x a baked per-tier shape factor
                              # (all factors 1.0 = carry-forward; not
                              # shipped -- cumulative GTA V rarity is the
                              # wrong object for a level prior).

Override for local experiments only: MAKEFUN_MODEL=<name>. The grader runs
with no environment, so the in-file MODEL is what ships.

Spread (v2): the quantiles are a BIMODAL mixture, not a symmetric band.
The shipped model builds two scenarios per trophy -- no-refresh (P flat,
earners grow; spread covers random walk + a fraction of the upside headroom
to the E_inf cap rate) and refresh (P x R; spread adds the refresh-size
uncertainty + an OVERDUE term that GROWS with the time since the last
refresh, because a long flat stretch means a refresh is overdue). The lower
quantiles come from the refresh mass, the upper quantiles from earner
growth. The non-shipped models use spread_sigma (random walk + overdue +
trend). The v1 staleness term is deleted: it narrowed the band exactly when
a refresh was most overdue, which is backwards. On the flat practice
histories every model collapses to a near-carry-forward median (no movement
to fit, and the refresh probability over a 90-min horizon is small); only
the band differs from the house baseline, which is exactly the point: a
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

# --- shared spread parameters (FINAL v2, 2026-11-19; see docs/MODEL-DECISION.md)
# The spread is a BIMODAL mixture (refresh downside + earner upside), not a
# symmetric band. The shipped model builds its quantiles from two scenarios
# (no-refresh / refresh); the non-shipped models use spread_sigma below.
SIGMA_RW = 0.5          # logit units per sqrt(day), house baseline's value.
SIGMA_CAP = 2.5         # hard cap on total logit spread.
SIGMA_FLOOR = 0.05      # never report a (near-)degenerate band.

# --- refresh-event (denominator) model -- v2: P(t) as a scenario mixture ---
# The tracked-player denominator moves by REFRESH EVENTS (steps), not smooth
# drift: the launch-night refresh took 638 -> 3,868 players (x6.06) and cut
# every rate by x0.165. A log-linear P(t) fit is the wrong family for a step
# process, so the v2 model treats P as an event process: a refresh probability
# p before the target and a refresh size R. The quantiles are the mixture of
# a no-refresh scenario (P flat) and a refresh scenario (P x R). The median
# shifts down when a refresh is likely and does not dilute when none is
# expected. P_SLOPE_CAP (v1) is DELETED: the explicit scenario weight
# replaces the uncalibrated slope cap.
REFRESH_JUMP_RATIO = 1.5   # a P jump >= 1.5x between consecutive reads is a
                           # refresh event (smooth growth between 10-30 min
                           # reads is <1%; the observed jump is 6.06x).
PRIOR_REFRESH_RATE = 1.0 / 24.0  # prior MEAN refresh rate, per hour (1/day).
                           # The prior is Gamma(shape=ALPHA, rate=BETA_H),
                           # so the posterior mean after n jumps in a span of
                           # T hours is (ALPHA+n)/(BETA_H+T): a FLAT history
                           # LOWERS the rate (evidence against refreshes) --
                           # 0 jumps in 48h -> 1/72 per hour (0.33/day), not
                           # 1/24. STATED ASSUMPTION: the one documented
                           # refresh fell inside the first day; a stats site
                           # refreshing daily is the standard cadence. No
                           # launch-week refresh cadence exists to estimate
                           # this from (Wayback 403; see docs).
PRIOR_REFRESH_ALPHA = 1.0  # Gamma shape of the refresh-rate prior
PRIOR_REFRESH_BETA_H = 24.0  # Gamma rate denominator (hours): prior mean 1/24
PRIOR_REFRESH_SIZE = 3868.0 - 638.0  # 3230 players: the documented
                           # launch-night refresh ADDED this many tracked
                           # players. A DATA point (RULES.txt), not an
                           # assumption. The refresh size is modelled as an
                           # ADDITION (players newly tracked), not a ratio:
                           # a ratio compounds absurdly (a second 6x jump
                           # would take P to ~23k), while the number of
                           # players a refresh newly tracks is roughly the
                           # same order each time. The ratio R for a scenario
                           # is (P_last + size) / P_last, so it shrinks as P
                           # grows -- a second refresh dilutes far less than
                           # the launch-night catch-up did.
P_REFRESH_MAX = 0.95       # cap on the refresh probability: keep a small
                           # no-refresh mass so the upper quantiles retain
                           # the earner-growth scenario.
SIZE_SIGMA = 0.5           # logit spread of the refresh SIZE inside the
                           # refresh scenario (the next refresh's size is
                           # uncertain; one observed size is one draw).
OVERDUE_SIGMA = 1.0        # max spread added when a refresh is OVERDUE.
OVERDUE_TAU_HOURS = 24.0   # overdue saturation time.
HEADROOM_FRAC = 0.25       # fraction of the upside headroom (logit gap from
                           # the last rate to the E_inf cap rate) covered by
                           # the no-refresh scenario's spread, so the upper
                           # quantiles can reach into the headroom.

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


def norm_cdf(z: float) -> float:
    """Standard normal CDF (via erf)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _symmetric_qlgs(median_lg: float, sigma: float, levels):
    """Quantile logits of a symmetric normal centred on median_lg."""
    return [median_lg + norm_ppf(lv) * sigma for lv in levels]


def _mixture_qlgs(mu_a: float, sig_a: float, mu_b: float, sig_b: float,
                  p: float, levels):
    """Quantile logits of a two-component normal mixture on the logit scale:
    component A (no refresh) with weight 1-p, component B (refresh) with
    weight p. Solved by bisection on the mixture CDF
    F(x) = (1-p)*Phi((x-mu_a)/sig_a) + p*Phi((x-mu_b)/sig_b).
    The result is BIMODAL in the right way: the lower quantiles come from
    the refresh mass, the upper quantiles from the no-refresh mass."""
    lo = min(mu_a, mu_b) - 8.0 * max(sig_a, sig_b)
    hi = max(mu_a, mu_b) + 8.0 * max(sig_a, sig_b)
    out = []
    for lv in levels:
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            f = (1.0 - p) * norm_cdf((m - mu_a) / sig_a) + \
                p * norm_cdf((m - mu_b) / sig_b)
            if f < lv:
                a = m
            else:
                b = m
        out.append(0.5 * (a + b))
    return out


def detect_refreshes(hist, release):
    """Find refresh events in the players-tracked series: a STEP, not smooth
    growth. A step is a jump of >= REFRESH_JUMP_RATIO between consecutive
    reads that is ALSO much larger than the typical step (an outlier in the
    log-increments). The typical-step test matters: smooth linear growth
    from a small base looks like a big ratio jump (e.g. 638 -> 1038 is
    x1.63) but is not a refresh. Returns a list of (hours_since_release,
    ratio)."""
    pts = []
    for h in hist:
        p = h.get("players_tracked")
        if not p:
            continue
        t = (parse_time(h["at"]) - release).total_seconds() / 3600.0
        pts.append((t, float(p)))
    if len(pts) < 2:
        return []
    logsteps = [math.log(pts[i][1] / pts[i - 1][1])
                for i in range(1, len(pts)) if pts[i - 1][1] > 0]
    abssteps = sorted(abs(x) for x in logsteps)
    med = abssteps[len(abssteps) // 2] if abssteps else 0.0
    jumps = []
    for i in range(1, len(pts)):
        if pts[i - 1][1] <= 0:
            continue
        r = pts[i][1] / pts[i - 1][1]
        lr = math.log(r)
        if r >= REFRESH_JUMP_RATIO and lr >= 5.0 * max(med, 1e-9):
            jumps.append((pts[i][0], r, pts[i][1] - pts[i - 1][1]))
    return jumps


def refresh_params(hist, horizon_h, t_at, release):
    """Denominator event process: refresh probability p before the target,
    refresh size (players ADDED by a refresh), hours since the last
    refresh (gap), number of observed jumps, and E[N | N>=1] -- the
    expected number of refreshes before the target GIVEN at least one
    occurs (mu / (1 - exp(-mu)), mu = rate * horizon). The refresh
    scenario's denominator is P_last + E[N|N>=1] * size, NOT P_last +
    size: over a multi-day horizon several refreshes are expected, and
    modelling only one UNDER-dilutes (the simulator caught this: the
    single-refresh mixture lost to the old capped log-linear model even
    when the refresh process matched the prior). The rate is the Gamma
    posterior mean (ALPHA+n)/(BETA_H+span) -- a flat history lowers it.
    The size is the median observed jump ADDITION, else the documented
    launch-night addition (a DATA point)."""
    jumps = detect_refreshes(hist, release)
    t_first = (parse_time(hist[0]["at"]) - release).total_seconds() / 3600.0
    t_last = (parse_time(hist[-1]["at"]) - release).total_seconds() / 3600.0
    span = max(1.0, t_last - t_first)
    rate = (PRIOR_REFRESH_ALPHA + len(jumps)) / (PRIOR_REFRESH_BETA_H + span)
    if jumps:
        adds = sorted(a for (_, _, a) in jumps)
        size = adds[len(adds) // 2]
        last_jump_t = jumps[-1][0]
    else:
        size = PRIOR_REFRESH_SIZE
        last_jump_t = None
    mu = rate * max(0.0, horizon_h)
    p = 1.0 - math.exp(-mu)
    p = min(P_REFRESH_MAX, max(0.0, p))
    n_given = mu / p if p > 1e-9 else 1.0
    if last_jump_t is None:
        gap = max(0.0, t_at - t_first)
    else:
        gap = max(0.0, t_at - last_jump_t)
    return p, size, gap, len(jumps), n_given


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
    """Shared logit spread for the non-shipped models: random walk +
    refresh-overdue + trend uncertainty. The overdue term GROWS with the
    time since the trophy's rate last changed -- a long flat stretch means
    a refresh is overdue, so the band WIDENS. The v1 staleness term had
    this backwards (it narrowed the band exactly when a refresh was most
    overdue); it is deleted."""
    days = horizon_h / 24.0
    rw = SIGMA_RW * math.sqrt(days)
    gap = series.last_change_gap_h
    if gap is None:
        overdue = OVERDUE_SIGMA
    else:
        overdue = OVERDUE_SIGMA * (1.0 - math.exp(-gap / OVERDUE_TAU_HOURS))
    trend = abs(se_slope) * horizon_h
    sigma = math.sqrt(rw * rw + overdue * overdue + trend * trend)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


# ---------------------------------------------------------------------------
# The three candidate model families. Each returns (median_logit, sigma).
# ---------------------------------------------------------------------------
def model_logit_linear(case, series, horizon_h, t0):
    """Not shipped (see MODEL): shrunk OLS of logit(rate) on hours over the last
    FIT_WINDOW_HOURS of reads, extrapolated to the target. Flat history ->
    slope 0 -> carry-forward median. Returns the quantile logits (symmetric)."""
    levels = case.get("levels", LEVELS)
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
        return _symmetric_qlgs(median_lg,
                               spread_sigma(case, series, horizon_h, se_slope=se_b),
                               levels)
    if series.last_rate is not None:
        return _symmetric_qlgs(logit(series.last_rate),
                               spread_sigma(case, series, horizon_h), levels)
    return _symmetric_qlgs(logit(1.0), SIGMA_CAP, levels)  # no observations
    # at all: wide, centred low


def model_poisson_hazard(case, series, horizon_h, t0):
    """Poisson-hazard fit on earner COUNTS + denominator as a SCENARIO
    MIXTURE (v2 -- see docs/MODEL-DECISION.md). E(t) = E_inf * (1 - exp(-lam*t)),
    lam picked by penalized least squares over a grid, E_inf by linear scale
    at that lam, with the GTA V prior (hard E_inf cap + weak log-lambda
    penalty). The denominator is NOT fit log-linearly (that is the wrong
    family for a step process, and the v1 P_SLOPE_CAP moved the median by
    ~1 logit through an uncalibrated constant): instead the refresh
    probability p and size R come from refresh_params(), and the quantiles
    are the mixture of a no-refresh scenario (P flat, earners grow) and a
    refresh scenario (P + refresh_add players, earners grow) -- BIMODAL, lower quantiles from
    the refresh mass, upper quantiles from earner growth. The median shifts
    down when a refresh is likely and does not dilute when none is expected.
    The refresh scenario's spread adds the refresh-size uncertainty and the
    OVERDUE term (a long flat stretch widens the band -- the v1 staleness
    term had this backwards). The no-refresh scenario's spread covers a
    fraction of the upside headroom to the E_inf cap rate. Flat history ->
    the earners fit saturates and, on a short horizon, p is small, so the
    median stays near carry-forward. Returns the quantile logits."""
    release = parse_time(case.get("game", {}).get("release_date", case["history"][0]["at"]))
    hist = case["history"]
    levels = case.get("levels", LEVELS)
    if series.last_rate is None or not hist:
        return _symmetric_qlgs(logit(1.0), SIGMA_CAP, levels)
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
        return _symmetric_qlgs(logit(series.last_rate),
                               spread_sigma(case, series, horizon_h), levels)
    # Denominator event process (scenario mixture -- replaces log-linear P(t)).
    t_target = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    t_at = (parse_time(case["at"]) - release).total_seconds() / 3600.0
    p_last = float(hist[-1]["players_tracked"])
    p_refresh, refresh_add, gap_refresh, _nj, n_given = refresh_params(
        hist, horizon_h, t_at, release)
    # Refresh ratio for this trophy's P_last: the refresh scenario expects
    # E[N|N>=1] refreshes before the target, each ADDING refresh_add
    # players, so the ratio shrinks as P grows (a second refresh dilutes
    # far less than the launch-night catch-up did).
    refresh_ratio = (p_last + n_given * refresh_add) / max(p_last, 1.0)
    # E_inf cap is a rate cap at the NO-refresh target (P flat): the earners
    # can grow up to the cap regardless of which denominator scenario holds.
    # Using P_last is the binding (conservative) form of the cap.
    e_inf_cap = max(obs[-1][1], E_INF_CAP_MULT * sat * p_last)
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
        return _symmetric_qlgs(logit(series.last_rate),
                               spread_sigma(case, series, horizon_h), levels)
    _, lam, e_inf = best
    e_target = e_inf * (1.0 - math.exp(-lam * max(t_target, 0.0)))
    # --- scenario A: no refresh (P flat) ----------------------------------
    rate_a = clamp(100.0 * max(e_target, 0.0) / max(p_last, 1.0))
    mu_a = logit(rate_a)
    # Upside headroom: the earners can grow up to the cap rate.
    cap_rate = clamp(100.0 * e_inf_cap / max(p_last, 1.0))
    headroom = max(0.0, logit(cap_rate) - logit(series.last_rate))
    rw = SIGMA_RW * math.sqrt(horizon_h / 24.0)
    sig_a = math.sqrt(rw * rw + (HEADROOM_FRAC * headroom) ** 2)
    sig_a = min(SIGMA_CAP, max(SIGMA_FLOOR, sig_a))
    # --- scenario B: refresh (P + refresh_add) -----------------------------
    rate_b = clamp(100.0 * max(e_target, 0.0) / max(p_last * refresh_ratio, 1.0))
    mu_b = logit(rate_b)
    overdue = OVERDUE_SIGMA * (1.0 - math.exp(-gap_refresh / OVERDUE_TAU_HOURS))
    sig_b = math.sqrt(sig_a * sig_a + SIZE_SIGMA * SIZE_SIGMA + overdue * overdue)
    sig_b = min(SIGMA_CAP, max(SIGMA_FLOOR, sig_b))
    return _mixture_qlgs(mu_a, sig_a, mu_b, sig_b, p_refresh, levels)


def model_prior_shape(case, series, horizon_h, t0):
    """Baked shape prior: last rate x per-tier factor. All factors are 1.0
    (carry-forward) because no credible week-one launch curve exists to bake
    in -- cumulative GTA IV/V rarities are long-run equilibria, the wrong
    object for a 4.5-day forecast. See docs/MODEL-DECISION.md. Returns the
    quantile logits (symmetric)."""
    levels = case.get("levels", LEVELS)
    tier = case["trophies"][series.idx].get("tier", "bronze")
    factor = SHAPE_FACTOR.get(tier, 1.0)
    if series.last_rate is not None:
        return _symmetric_qlgs(logit(clamp(series.last_rate * factor)),
                               spread_sigma(case, series, horizon_h), levels)
    return _symmetric_qlgs(logit(1.0), SIGMA_CAP, levels)


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
            qlgs = _symmetric_qlgs(logit(fallback_rate), SIGMA_CAP, levels)
        else:
            qlgs = fn(case, s, horizon_h, t0)
        qs = []
        prev = None
        for lg in qlgs:
            q = round4(clamp(sigmoid(lg), 0.0, 100.0))
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
