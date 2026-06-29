"""
app.py — Bowler Coaching Report Tool  (single-file, self-contained)

Two views, shown as tabs:
  1. Performance Fingerprint — a bowler's economy / strike-rate / dot-% /
     boundary-% as percentiles vs a peer cohort.
  2. Role Identification — k-means learns three bowling archetypes (Powerplay
     strike / Middle-overs control / Death specialist) from how well each bowler
     performs in every phase, then tests whether he is actually *bowled* in the
     phase his skill profile suits.

Run with:
    streamlit run app.py

Expects a cleaned ball-by-ball file produced by clean_t20_deliveries.py:
    t20_deliveries_clean.parquet   (preferred, fast)
    t20_deliveries_clean.csv       (fallback)

Cricket rules applied (so the numbers are correct, not just plausible):
  * Legal ball       -> not a wide and not a no-ball.
  * Runs charged to the bowler -> batter runs + wides + no-balls
                                   (byes / leg-byes / penalty are NOT his fault).
  * Bowler's wicket  -> is_wicket == 1 AND wicket_kind is a bowler-credited mode
                        (run-outs, retired, obstructing the field do NOT count).
  * Dot ball         -> a legal delivery on which zero runs are scored.
  * Boundary conceded-> four or six off the bat (boundary byes don't count).
  * Phase            -> Powerplay overs 1-6, Middle 7-15, Death 16-20.
                        The `over` column is 0-indexed, so 0-5 / 6-14 / 15-19.

UI behaviour:
  * Filters live inside a form per tab — nothing recomputes until you press
    "Apply filters", so the view never refreshes on its own while you edit.
  * The selected bowler is preserved when you change filters; it only resets if
    that bowler drops out of the current cohort.
"""

import importlib.util
import io
import math
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - friendlier error surfaced in the UI
    _HAVE_SKLEARN = False

try:
    # Only used to give the trend/step tests an exact p-value; everything still
    # works without SciPy via a normal/Student-t approximation below.
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - graceful fallback when SciPy is absent
    _HAVE_SCIPY = False

# matplotlib is only needed to render the downloadable PDF report; the app runs
# fine without it (the export button just explains how to enable it).
_HAVE_MPL = importlib.util.find_spec("matplotlib") is not None

# --------------------------------------------------------------------------- #
# Config & metric definitions
# --------------------------------------------------------------------------- #
CLEAN_PARQUET = "t20_deliveries_clean.parquet"
# CLEAN_CSV = "t20_deliveries_clean.csv" 

USECOLS = [
    "match_id", "innings", "over", "ball", "bowler",
    "match_date", "bowling_team", "batting_team",
    "runs_batter", "runs_extras", "runs_total",
    "extra_wides", "extra_noballs", "extra_byes", "extra_legbyes", "extra_penalty",
    "is_wicket", "wicket_kind",
]

# Wicket kinds that are credited to the bowler.
BOWLER_WICKET_KINDS = {
    "bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket",
}

# Phase definition using the 0-indexed `over` column.
PHASES = {
    "Powerplay": range(0, 6),    # overs 1-6
    "Middle":    range(6, 15),   # overs 7-15
    "Death":     range(15, 20),  # overs 16-20
}

# Metric metadata: label, and whether a lower raw value is better.
# `lower_is_better` drives the percentile flip so high pct = good everywhere.
FINGERPRINT_METRICS = {
    "economy":      {"label": "Economy",      "lower_is_better": True,  "unit": "rpo"},
    "strike_rate":  {"label": "Strike rate",  "lower_is_better": True,  "unit": "balls/wkt"},
    "dot_pct":      {"label": "Dot-ball %",   "lower_is_better": False, "unit": "%"},
    "boundary_pct": {"label": "Boundary %",   "lower_is_better": True,  "unit": "%"},
}

# --- Role Identification config -------------------------------------------- #
# Canonical phase order for every phase-keyed frame in the role logic.
ROLE_PHASE_ORDER = ["Powerplay", "Middle", "Death"]

# Human-readable archetype names, one per phase.
ROLE_ARCHETYPE = {
    "Powerplay": "Powerplay strike",
    "Middle": "Middle-overs control",
    "Death": "Death specialist",
}

# Per-phase skill metrics and whether a lower raw value is better (so we flip the
# percentile and high skill always = good).
ROLE_SKILL_METRICS = {
    "economy":      True,   # lower economy    = better
    "strike_rate":  True,   # lower strike rate = better
    "dot_pct":      False,  # higher dot %      = better
    "boundary_pct": True,   # lower boundary %  = better
}

PHASE_COLORS = {"Powerplay": "#2E86DE", "Middle": "#27AE60", "Death": "#E67E22"}

# --- Development trajectory config ------------------------------------------ #
# Metrics tracked across a bowler's career timeline. Each carries its display
# label, the direction that counts as *improvement*, a value formatter, and the
# (numerator, denominator) count columns the rolling rate is rebuilt from — a
# rolling window is therefore a true pooled rate over the window, not an average
# of noisy per-match rates.
TRAJECTORY_METRICS = {
    "economy":      {"label": "Economy",     "lower_is_better": True,  "fmt": "{:.2f}", "unit": "rpo",
                     "counts": ("runs", "balls")},
    "dot_pct":      {"label": "Dot-ball %",  "lower_is_better": False, "fmt": "{:.1f}", "unit": "%",
                     "counts": ("dots", "balls")},
    "boundary_pct": {"label": "Boundary %",  "lower_is_better": True,  "fmt": "{:.1f}", "unit": "%",
                     "counts": ("boundaries", "balls")},
    "strike_rate":  {"label": "Strike rate", "lower_is_better": True,  "fmt": "{:.1f}", "unit": "balls/wkt",
                     "counts": ("balls", "wickets")},
}


# --------------------------------------------------------------------------- #
# Metrics logic
# --------------------------------------------------------------------------- #
def _phase_of(over: pd.Series) -> pd.Series:
    out = pd.Series("Middle", index=over.index, dtype="object")
    out[over < 6] = "Powerplay"
    out[over >= 15] = "Death"
    return out


def add_ball_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-delivery helper columns used by the aggregation."""
    d = df.copy()

    for c in ["runs_batter", "runs_extras", "runs_total",
              "extra_wides", "extra_noballs", "extra_byes",
              "extra_legbyes", "extra_penalty", "is_wicket", "over"]:
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    # Legal delivery: not a wide, not a no-ball.
    d["is_legal"] = (d["extra_wides"] == 0) & (d["extra_noballs"] == 0)

    # Runs the bowler is accountable for (excludes byes/leg-byes/penalty).
    d["runs_off_bowler"] = d["runs_batter"] + d["extra_wides"] + d["extra_noballs"]

    # Bowler-credited wicket.
    kind = d.get("wicket_kind", pd.Series(index=d.index, dtype="object"))
    kind = kind.astype("string").str.lower().str.strip()
    d["bowler_wicket"] = ((d["is_wicket"] == 1) & kind.isin(BOWLER_WICKET_KINDS)).astype(int)

    # Dot ball: legal delivery, zero runs scored on it.
    d["is_dot"] = (d["is_legal"] & (d["runs_total"] == 0)).astype(int)

    # Boundary off the bat.
    d["is_boundary"] = d["runs_batter"].isin([4, 6]).astype(int)

    d["phase"] = _phase_of(d["over"])
    return d


def aggregate_bowlers(df: pd.DataFrame, phase: str | None = None) -> pd.DataFrame:
    """Aggregate to one row per bowler and compute the four metrics."""
    if "is_legal" not in df.columns:
        df = add_ball_flags(df)
    if phase is not None:
        df = df[df["phase"] == phase]

    g = df.groupby("bowler", dropna=True)
    agg = g.agg(
        innings=("match_id", "nunique"),
        balls_legal=("is_legal", "sum"),
        runs=("runs_off_bowler", "sum"),
        wickets=("bowler_wicket", "sum"),
        dots=("is_dot", "sum"),
        boundaries=("is_boundary", "sum"),
    )

    legal = agg["balls_legal"].replace(0, np.nan)
    overs = legal / 6.0

    agg["economy"] = agg["runs"] / overs
    agg["strike_rate"] = legal / agg["wickets"].replace(0, np.nan)  # NaN if 0 wkts
    agg["dot_pct"] = 100 * agg["dots"] / legal
    agg["boundary_pct"] = 100 * agg["boundaries"] / legal
    agg["overs"] = overs.round(1)
    return agg


def add_percentiles(agg: pd.DataFrame, metrics: list[str] | None = None,
                    min_balls: int = 60) -> pd.DataFrame:
    """Add a *_pctile column per metric vs the peer cohort (>= min_balls).
    Lower-is-better metrics are flipped so high percentile always = good."""
    metrics = metrics or list(FINGERPRINT_METRICS)
    out = agg.copy()
    cohort = out[out["balls_legal"] >= min_balls]

    for m in metrics:
        col = cohort[m].dropna()
        pct_col = f"{m}_pctile"
        if col.empty:
            out[pct_col] = np.nan
            continue
        ranks = col.rank(pct=True) * 100
        if FINGERPRINT_METRICS[m]["lower_is_better"]:
            ranks = 100 - ranks
        out[pct_col] = ranks

    out["in_cohort"] = out["balls_legal"] >= min_balls
    return out


# --------------------------------------------------------------------------- #
# Role Identification logic
#
# Method
# ------
# 1. Split every bowler's deliveries into the three phases and compute phase-level
#    economy, strike rate, dot-% and boundary-%.
# 2. Turn each phase metric into a percentile within the cohort that bowled enough
#    balls in that phase (lower-is-better metrics flipped so high = good), and
#    average them to a single 0-100 "phase skill" per phase.
# 3. A phase the bowler never bowls (or barely bowls) counts as 0, so his 3-D
#    skill vector points along the axis of the phase(s) he actually bowls and
#    excels in — this keeps the three k-means clusters cleanly aligned to phases.
# 4. k-means (k = 3) on the standardised skill vector groups similar bowlers; each
#    cluster is named after the phase its centroid peaks on, via a unique greedy
#    assignment, giving exactly the three archetypes.
# 5. Deployment = the phase he actually bowls the most legal balls in. If the
#    archetype phase != deployment phase, he may be mis-used.
# --------------------------------------------------------------------------- #
def phase_bowler_table(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per (bowler, phase) with balls and the four core rates."""
    g = flagged.groupby(["bowler", "phase"], dropna=True)
    agg = g.agg(
        balls=("is_legal", "sum"),
        runs=("runs_off_bowler", "sum"),
        wickets=("bowler_wicket", "sum"),
        dots=("is_dot", "sum"),
        boundaries=("is_boundary", "sum"),
    ).reset_index()

    balls = agg["balls"].replace(0, np.nan)
    agg["economy"] = agg["runs"] / (balls / 6.0)
    agg["strike_rate"] = balls / agg["wickets"].replace(0, np.nan)  # NaN if 0 wkts
    agg["dot_pct"] = 100 * agg["dots"] / balls
    agg["boundary_pct"] = 100 * agg["boundaries"] / balls
    return agg


def phase_skill_matrix(flagged: pd.DataFrame,
                       min_phase_balls: int = 30) -> pd.DataFrame:
    """Bowler × phase matrix of 0-100 'phase skill' scores.

    For each phase we rank the bowlers who bowled >= min_phase_balls in it on the
    four metrics (lower-is-better metrics flipped so high = good), average the
    available percentiles, and leave phases without a meaningful sample as NaN
    (filled with 0 before clustering)."""
    pbt = phase_bowler_table(flagged)
    pieces = []

    for phase in ROLE_PHASE_ORDER:
        sub = pbt[(pbt["phase"] == phase) & (pbt["balls"] >= min_phase_balls)].copy()
        if sub.empty:
            continue
        pct_cols = []
        for metric, lower_is_better in ROLE_SKILL_METRICS.items():
            ranks = sub[metric].rank(pct=True) * 100
            if lower_is_better:
                ranks = 100 - ranks
            pct_cols.append(ranks)
        # Average the percentiles that exist (a no-wicket bowler has NaN strike
        # rate; we simply skip it rather than penalise or reward him for it).
        sub["skill"] = pd.concat(pct_cols, axis=1).mean(axis=1, skipna=True)
        pieces.append(sub[["bowler", "phase", "skill"]])

    if not pieces:
        return pd.DataFrame(columns=ROLE_PHASE_ORDER)

    skill = (
        pd.concat(pieces)
        .pivot(index="bowler", columns="phase", values="skill")
        .reindex(columns=ROLE_PHASE_ORDER)
    )
    return skill


def deployment_shares(flagged: pd.DataFrame) -> pd.DataFrame:
    """Bowler × phase matrix: % of a bowler's legal balls bowled in each phase."""
    legal = flagged[flagged["is_legal"].astype(bool)]
    counts = (
        legal.groupby(["bowler", "phase"]).size().unstack(fill_value=0)
        .reindex(columns=ROLE_PHASE_ORDER, fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0) * 100.0


def _assign_unique_phases(centroids: pd.DataFrame) -> dict:
    """Map each cluster to a *distinct* phase.

    With k = 3 clusters and 3 phases we want exactly one cluster per archetype.
    A naive idxmax can map two clusters to the same phase, so we greedily take the
    strongest (cluster, phase) pairing first and lock both out, repeating until
    every cluster and phase is used."""
    pairs = [
        (centroids.loc[c, p], c, p)
        for c in centroids.index
        for p in ROLE_PHASE_ORDER
    ]
    pairs.sort(reverse=True)  # strongest centroid-on-phase first

    cluster_to_phase: dict = {}
    used_phases: set = set()
    for _, c, p in pairs:
        if c in cluster_to_phase or p in used_phases:
            continue
        cluster_to_phase[c] = p
        used_phases.add(p)
    # If k < number of phases, any leftover clusters fall back to their idxmax.
    for c in centroids.index:
        cluster_to_phase.setdefault(c, centroids.loc[c].idxmax())
    return cluster_to_phase


def identify_roles(flagged: pd.DataFrame,
                   min_total_balls: int = 120,
                   min_phase_balls: int = 30,
                   k: int = 3,
                   random_state: int = 42) -> pd.DataFrame | None:
    """Cluster bowlers into role archetypes and test their deployment.

    Returns a DataFrame indexed by bowler (Powerplay/Middle/Death skills,
    *_share deployment shares, cluster, archetype_phase, archetype,
    deployment_phase, in_role, total_balls) or None if there isn't enough data."""
    if not _HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is required for role identification. "
            "Install it with: pip install scikit-learn"
        )

    skill = phase_skill_matrix(flagged, min_phase_balls=min_phase_balls)
    if skill.empty:
        return None

    # Eligibility: enough total legal balls to be worth profiling at all.
    total_balls = (
        flagged[flagged["is_legal"].astype(bool)].groupby("bowler").size()
    )
    eligible = total_balls[total_balls >= min_total_balls].index
    skill = skill.loc[skill.index.intersection(eligible)]
    if len(skill) < k:
        return None

    # Phases the bowler never (meaningfully) bowled count as 0 skill, so the
    # vector points along the axis of the phase(s) he actually bowls.
    feat = skill.fillna(0.0)
    X = StandardScaler().fit_transform(feat[ROLE_PHASE_ORDER].values)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    out = feat.copy()
    out["cluster"] = labels

    centroids = out.groupby("cluster")[ROLE_PHASE_ORDER].mean()
    cluster_to_phase = _assign_unique_phases(centroids)
    out["archetype_phase"] = out["cluster"].map(cluster_to_phase)
    out["archetype"] = out["archetype_phase"].map(ROLE_ARCHETYPE)

    # Deployment: where the balls actually land.
    shares = deployment_shares(flagged).reindex(out.index)
    out["deployment_phase"] = shares[ROLE_PHASE_ORDER].idxmax(axis=1)
    for p in ROLE_PHASE_ORDER:
        out[f"{p}_share"] = shares[p]

    out["in_role"] = out["archetype_phase"] == out["deployment_phase"]
    out["total_balls"] = total_balls.reindex(out.index)

    return out


def cluster_summary(roles: pd.DataFrame) -> pd.DataFrame:
    """Per-archetype summary: count, mean phase skills, % deployed in role."""
    if roles is None or roles.empty:
        return pd.DataFrame()
    g = roles.groupby("archetype")
    summary = g.agg(
        bowlers=("archetype", "size"),
        Powerplay=("Powerplay", "mean"),
        Middle=("Middle", "mean"),
        Death=("Death", "mean"),
        pct_in_role=("in_role", "mean"),
    )
    summary["pct_in_role"] = (summary["pct_in_role"] * 100).round(0)
    order = [ROLE_ARCHETYPE[p] for p in ROLE_PHASE_ORDER if ROLE_ARCHETYPE[p] in summary.index]
    return summary.reindex(order)


# --------------------------------------------------------------------------- #
# Execution Consistency & Matchup logic
#
# Two ideas live here, both built straight off the flagged ball-by-ball frame:
#
#  1. Execution consistency — collapse every delivery into the over it belongs to
#     (match × innings × over × bowler) and sum the runs the bowler is charged.
#     Each over a bowler sends down is then one observation of "runs conceded in
#     an over". The spread of that distribution (its standard deviation) is how
#     repeatable he is: 6,7,6,7 is a tighter, more coachable groove than
#     0,15,2,14 even when both average ~7. We rank low spread high, so a high
#     consistency percentile = a bowler who reproduces the same over again and
#     again — and the gap to the cohort's tightest bowlers is a concrete drill
#     target.
#
#  2. Matchup & dismissal profile — two tactical lenses on the same balls:
#       * Dismissal profile: the mix of *how* his wickets fall (bowled, caught,
#         lbw, …) versus how the peer cohort's wickets fall.
#       * Runs leak: every run he concedes split into boundaries (4s/6s off the
#         bat), runs taken in ones/twos/threes, and extras (wides + no-balls),
#         so you can see whether he bleeds through the rope or the gaps.
#       * Opposition matchup: economy and wickets broken down by batting team,
#         surfacing who he dominates and who gets after him.
# --------------------------------------------------------------------------- #
def over_runs_table(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per over a bowler bowled, with the runs charged to him in it.

    An "over" here is a (match, innings, over#, bowler) group — that last key
    keeps a mid-over bowling change as two separate partial overs rather than
    fusing them, which is the honest unit for measuring repeatability."""
    if "runs_off_bowler" not in flagged.columns:
        flagged = add_ball_flags(flagged)
    g = flagged.groupby(["match_id", "innings", "over", "bowler"], dropna=True)
    ot = g.agg(
        over_runs=("runs_off_bowler", "sum"),
        legal_balls=("is_legal", "sum"),
    ).reset_index()
    # Phase of each over, so the histogram/buckets can be read per phase too.
    ot["phase"] = _phase_of(ot["over"])
    return ot


def consistency_table(over_runs: pd.DataFrame, min_overs: int = 20) -> pd.DataFrame:
    """One row per bowler: mean / spread of his runs-per-over and a 0–100
    consistency percentile (high = tightest spread vs the cohort)."""
    g = over_runs.groupby("bowler")["over_runs"]
    s = g.agg(overs_bowled="count", mean_runs="mean", std_runs="std",
              max_runs="max", min_runs="min").copy()
    # A bowler with one over has NaN std; treat the spread as undefined.
    s["cv"] = s["std_runs"] / s["mean_runs"].replace(0, np.nan)

    cohort = s[s["overs_bowled"] >= min_overs]
    s["consistency_pctile"] = np.nan
    if not cohort.empty:
        ranks = cohort["std_runs"].rank(pct=True) * 100  # low std ranks low...
        s.loc[cohort.index, "consistency_pctile"] = 100 - ranks  # ...flip: high = good
    s["in_cohort"] = s["overs_bowled"] >= min_overs
    return s


# Runs-per-over buckets used to show the *shape* of a bowler's overs.
OVER_BUCKETS = [
    ("Tight (0–4)",     lambda r: r <= 4),
    ("Par (5–8)",       lambda r: (r >= 5) & (r <= 8)),
    ("Costly (9–12)",   lambda r: (r >= 9) & (r <= 12)),
    ("Leaked (13+)",    lambda r: r >= 13),
]


def over_bucket_shares(over_runs_bowler: pd.Series) -> pd.DataFrame:
    """% of a bowler's overs landing in each runs-per-over band."""
    n = len(over_runs_bowler)
    rows = []
    for label, fn in OVER_BUCKETS:
        cnt = int(fn(over_runs_bowler).sum()) if n else 0
        rows.append({"band": label, "overs": cnt,
                     "pct": (100.0 * cnt / n) if n else 0.0})
    return pd.DataFrame(rows)


def dismissal_profile(flagged: pd.DataFrame, bowler: str | None = None) -> pd.Series:
    """Count of bowler-credited wickets by dismissal kind (optionally for one
    bowler; otherwise the whole cohort passed in)."""
    d = flagged[flagged["bowler_wicket"] == 1]
    if bowler is not None:
        d = d[d["bowler"] == bowler]
    kind = d.get("wicket_kind", pd.Series(dtype="object")).astype("string").str.lower().str.strip()
    return kind.value_counts()


def runs_leak_breakdown(flagged: pd.DataFrame, bowler: str) -> pd.DataFrame:
    """Split every run a bowler concedes into boundaries / running / extras."""
    d = flagged[flagged["bowler"] == bowler]
    is_bdry = d["runs_batter"].isin([4, 6])
    boundary_runs = int(d.loc[is_bdry, "runs_batter"].sum())
    running_runs = int(d.loc[~is_bdry, "runs_batter"].sum())
    extras_runs = int((d["extra_wides"] + d["extra_noballs"]).sum())
    total = boundary_runs + running_runs + extras_runs
    rows = [
        ("Boundaries (4s/6s)", boundary_runs),
        ("Running (1s/2s/3s)", running_runs),
        ("Extras (wd/nb)",     extras_runs),
    ]
    return pd.DataFrame(
        [{"source": s, "runs": r, "pct": (100.0 * r / total) if total else 0.0}
         for s, r in rows]
    )


def opposition_matchup(flagged: pd.DataFrame, bowler: str,
                       min_balls: int = 24) -> pd.DataFrame:
    """Per opposition batting-team economy / wickets for one bowler."""
    d = flagged[flagged["bowler"] == bowler]
    g = d.groupby("batting_team", dropna=True)
    agg = g.agg(
        balls_legal=("is_legal", "sum"),
        runs=("runs_off_bowler", "sum"),
        wickets=("bowler_wicket", "sum"),
        boundaries=("is_boundary", "sum"),
    )
    legal = agg["balls_legal"].replace(0, np.nan)
    agg["overs"] = (legal / 6.0).round(1)
    agg["economy"] = agg["runs"] / (legal / 6.0)
    agg = agg[agg["balls_legal"] >= min_balls]
    return agg.sort_values("economy")


# --------------------------------------------------------------------------- #
# Development trajectory logic
#
# The coaching question this answers: "we worked on his economy / his dots —
# is the work actually showing up over time, or are we just seeing noise?"
#
# Three steps, all built off the flagged ball-by-ball frame:
#
#  1. Timeline — collapse a bowler's career to one row per match in
#     chronological order, carrying the raw count columns (legal balls, runs,
#     dots, boundaries, wickets) so any rate can be rebuilt exactly.
#  2. Rolling metric — a trailing N-match window re-pools those counts into a
#     true rate (sum runs / sum overs, etc.), smoothing single-game noise so a
#     real direction can be seen.
#  3. Trend + change-points — an OLS slope (with a significance test) says
#     whether the rolling metric is drifting up or down, and a weighted
#     mean-shift change-point search finds the match around which the level
#     actually *stepped* — the fingerprint of an intervention that took, rather
#     than a slow drift. Every metric is read in its "improvement" direction so
#     "better" always means the coached change is showing up.
# --------------------------------------------------------------------------- #
def _t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t-statistic. Uses SciPy when present, else a
    normal approximation (fine for the df we see here)."""
    if not np.isfinite(t) or df <= 0:
        return np.nan
    if _HAVE_SCIPY:
        return float(2.0 * _scipy_stats.t.sf(abs(t), df))
    return float(math.erfc(abs(t) / math.sqrt(2.0)))  # normal approx of 2*sf


def bowler_match_timeline(frame: pd.DataFrame, bowler: str) -> pd.DataFrame:
    """One row per match a bowler played, in chronological order.

    Keeps the raw counts (legal balls, runs, dots, boundaries, wickets) plus the
    noisy per-match rates (used only as the faint underlay on the chart — the
    trend and change-points run off the re-pooled rolling counts instead)."""
    d = frame[frame["bowler"] == bowler]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("match_id", dropna=True)
    tl = g.agg(
        match_date=("match_date", "first"),
        balls=("is_legal", "sum"),
        runs=("runs_off_bowler", "sum"),
        wickets=("bowler_wicket", "sum"),
        dots=("is_dot", "sum"),
        boundaries=("is_boundary", "sum"),
    ).reset_index()

    tl["match_date"] = pd.to_datetime(tl["match_date"], errors="coerce")
    # Chronological; fall back to match_id when a date is missing or tied so the
    # order is always deterministic.
    tl = tl.sort_values(["match_date", "match_id"], na_position="last").reset_index(drop=True)
    tl = tl[tl["balls"] > 0].reset_index(drop=True)
    if tl.empty:
        return tl
    tl["seq"] = np.arange(1, len(tl) + 1)

    balls = tl["balls"].replace(0, np.nan)
    tl["economy"] = tl["runs"] / (balls / 6.0)
    tl["dot_pct"] = 100 * tl["dots"] / balls
    tl["boundary_pct"] = 100 * tl["boundaries"] / balls
    tl["strike_rate"] = balls / tl["wickets"].replace(0, np.nan)  # NaN when 0 wkts
    return tl


def rolling_metric(tl: pd.DataFrame, metric: str, window: int) -> pd.Series:
    """Trailing N-match rolling value of `metric`, re-pooled from raw counts.

    Summing the counts over the window and dividing once is far steadier than
    averaging per-match rates (a 1-over cameo can't swing it)."""
    num_col, den_col = TRAJECTORY_METRICS[metric]["counts"]
    num = tl[num_col].rolling(window, min_periods=window).sum()
    den = tl[den_col].rolling(window, min_periods=window).sum().replace(0, np.nan)
    if metric == "economy":
        return num / (den / 6.0)
    if metric == "strike_rate":
        return num / den
    return 100 * num / den


def pooled_metric(seg: pd.DataFrame, metric: str) -> float:
    """Single pooled value of `metric` over a slice of the timeline."""
    num_col, den_col = TRAJECTORY_METRICS[metric]["counts"]
    num = seg[num_col].sum()
    den = seg[den_col].sum()
    if den == 0:
        return np.nan
    if metric == "economy":
        return num / (den / 6.0)
    if metric == "strike_rate":
        return num / den
    return 100 * num / den


def linear_trend(y) -> dict | None:
    """OLS slope of a series on its own 0..n-1 index.

    Returns slope (per match), R², a two-sided p-value for the slope, and the
    fitted endpoints. NaNs are dropped first; None if fewer than 3 points."""
    s = pd.Series(y).dropna()
    n = len(s)
    if n < 3:
        return None
    x = np.arange(n, dtype=float)
    yv = s.to_numpy(dtype=float)
    xm, ym = x.mean(), yv.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0:
        return None
    slope = float(((x - xm) * (yv - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    yhat = intercept + slope * x
    ss_res = float(((yv - yhat) ** 2).sum())
    ss_tot = float(((yv - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = np.nan
    if n > 2:
        se = math.sqrt(ss_res / (n - 2) / sxx) if sxx > 0 else np.nan
        if se and se > 0:
            p = _t_two_sided_p(slope / se, n - 2)
    return {"slope": slope, "intercept": intercept, "r2": r2, "p": p, "n": n,
            "first_fit": float(yhat[0]), "last_fit": float(yhat[-1])}


def _weighted_cost(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted within-segment sum of squares about the weighted mean."""
    if len(y) == 0 or w.sum() == 0:
        return 0.0
    m = np.average(y, weights=w)
    return float((w * (y - m) ** 2).sum())


def detect_change_points(y, w, min_size: int = 3, max_cp: int = 3,
                         min_drop_frac: float = 0.08) -> list[int]:
    """Binary-segmentation change-point detection on a weighted mean-shift model.

    `y` is the per-match metric, `w` its ball-count weight (so a heavily-bowled
    match anchors the level more than a single over). Greedily splits the
    segment whose split most reduces the weighted SSE, stopping when no split
    buys at least `min_drop_frac` of the original spread or `max_cp` is hit.
    Returns the split indices (each the first index of a new segment), sorted."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    if n < 2 * min_size:
        return []
    total = _weighted_cost(y, w)
    if total <= 0:
        return []

    cps: list[int] = []
    segments = [(0, n)]
    for _ in range(max_cp):
        best = None  # (gain, idx, a, b)
        for (a, b) in segments:
            if b - a < 2 * min_size:
                continue
            base = _weighted_cost(y[a:b], w[a:b])
            for i in range(a + min_size, b - min_size + 1):
                cost = _weighted_cost(y[a:i], w[a:i]) + _weighted_cost(y[i:b], w[i:b])
                gain = base - cost
                if best is None or gain > best[0]:
                    best = (gain, i, a, b)
        if best is None or best[0] < min_drop_frac * total:
            break
        _, idx, a, b = best
        cps.append(idx)
        segments.remove((a, b))
        segments.extend([(a, idx), (idx, b)])
    return sorted(cps)


def segment_levels(cf: pd.DataFrame, metric: str, cps: list[int]) -> pd.DataFrame:
    """Pooled metric level (and date span) within each segment cut by `cps`.

    `cf` is the change-point working frame (timeline rows where the metric is
    defined), already reset to a 0..n-1 index so `cps` index straight into it."""
    bounds = [0] + list(cps) + [len(cf)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = cf.iloc[a:b]
        rows.append({
            "start_seq": int(seg["seq"].iloc[0]),
            "end_seq": int(seg["seq"].iloc[-1]),
            "start_date": seg["match_date"].iloc[0],
            "end_date": seg["match_date"].iloc[-1],
            "matches": int(len(seg)),
            "level": pooled_metric(seg, metric),
        })
    return pd.DataFrame(rows)


def changepoint_frame(tl: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Timeline rows where `metric` is defined (strike rate is NaN in no-wicket
    matches), reset to a clean 0..n-1 index for change-point work."""
    cf = tl[tl[metric].notna()].reset_index(drop=True)
    return cf


def _welch_p(y1, y2) -> float:
    """Two-sided Welch t-test p-value between two per-match samples."""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    n1, n2 = len(y1), len(y2)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = y1.var(ddof=1), y2.var(ddof=1)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return np.nan
    t = (y1.mean() - y2.mean()) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return _t_two_sided_p(t, df)


def trajectory_summary(tl: pd.DataFrame, window: int) -> pd.DataFrame:
    """One row per metric: career level, latest rolling level, trend direction
    and slope, and the most-recent detected step (before -> after)."""
    rows = []
    for metric, meta in TRAJECTORY_METRICS.items():
        lower_better = meta["lower_is_better"]
        career = pooled_metric(tl, metric)
        roll = rolling_metric(tl, metric, window).dropna()
        latest = float(roll.iloc[-1]) if not roll.empty else np.nan

        tr = linear_trend(roll)
        slope = tr["slope"] if tr else np.nan
        p_tr = tr["p"] if tr else np.nan
        # Improvement = moving in the good direction with a significant slope.
        if tr is None or not np.isfinite(slope):
            direction = "—"
        else:
            improving = (slope < 0) if lower_better else (slope > 0)
            if np.isfinite(p_tr) and p_tr < 0.05:
                direction = "Improving" if improving else "Worsening"
            else:
                direction = "Flat"

        # Most-recent step from the change-point search.
        cf = changepoint_frame(tl, metric)
        before = after = np.nan
        step_date = pd.NaT
        step_p = np.nan
        if len(cf) >= 6:
            cps = detect_change_points(cf[metric].to_numpy(),
                                       cf["balls"].to_numpy())
            if cps:
                segs = segment_levels(cf, metric, cps)
                before = float(segs["level"].iloc[-2])
                after = float(segs["level"].iloc[-1])
                step_date = segs["start_date"].iloc[-1]
                idx = cps[-1]
                step_p = _welch_p(cf[metric].iloc[:idx], cf[metric].iloc[idx:])

        rows.append({
            "metric": metric, "label": meta["label"],
            "career": career, "latest": latest,
            "slope_per10": slope * 10 if np.isfinite(slope) else np.nan,
            "trend_p": p_tr, "direction": direction,
            "step_before": before, "step_after": after,
            "step_date": step_date, "step_p": step_p,
        })
    return pd.DataFrame(rows).set_index("metric")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading & flagging deliveries…")
def load_flagged() -> pd.DataFrame:
    if Path(CLEAN_PARQUET).exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    elif Path(CLEAN_CSV).exists():
        df = pd.read_csv(CLEAN_CSV)
    else:
        st.error(
            f"Cleaned data not found. Run `python clean_t20_deliveries.py` first "
            f"to create `{CLEAN_PARQUET}` or `{CLEAN_CSV}`."
        )
        st.stop()
    keep = [c for c in USECOLS if c in df.columns]
    d = df[keep].copy()
    if "match_date" in d.columns:
        d["year"] = pd.to_datetime(d["match_date"], errors="coerce").dt.year
    else:
        d["year"] = np.nan
    if "bowling_team" not in d.columns:
        d["bowling_team"] = "Unknown"
    if "batting_team" not in d.columns:
        d["batting_team"] = "Unknown"
    return add_ball_flags(d)


@st.cache_data(show_spinner=False)
def filter_options() -> tuple[list[int], list[str]]:
    d = load_flagged()
    years = sorted(int(y) for y in d["year"].dropna().unique())
    countries = sorted(str(c) for c in d["bowling_team"].dropna().unique())
    return years, countries


@st.cache_data(show_spinner="Aggregating bowlers…")
def build_table(phase: str | None, min_balls: int,
                years: tuple[int, ...], countries: tuple[str, ...]) -> pd.DataFrame:
    d = load_flagged()
    if years:
        d = d[d["year"].isin(years)]
    if countries:
        d = d[d["bowling_team"].isin(countries)]
    agg = aggregate_bowlers(d, phase=phase)
    return add_percentiles(agg, min_balls=min_balls)


@st.cache_data(show_spinner="Clustering bowler roles…")
def build_roles(min_total_balls: int, min_phase_balls: int,
                years: tuple[int, ...], countries: tuple[str, ...]) -> pd.DataFrame | None:
    d = load_flagged()
    if years:
        d = d[d["year"].isin(years)]
    if countries:
        d = d[d["bowling_team"].isin(countries)]
    return identify_roles(d, min_total_balls=min_total_balls,
                          min_phase_balls=min_phase_balls)


@st.cache_data(show_spinner="Measuring over-by-over consistency…")
def build_consistency(min_overs: int, years: tuple[int, ...],
                      countries: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (consistency_table, over_runs_table) for the current filters."""
    d = load_flagged()
    if years:
        d = d[d["year"].isin(years)]
    if countries:
        d = d[d["bowling_team"].isin(countries)]
    ot = over_runs_table(d)
    cons = consistency_table(ot, min_overs=min_overs)
    return cons, ot


@st.cache_data(show_spinner="Building matchup & dismissal profile…")
def build_profile_frame(years: tuple[int, ...],
                        countries: tuple[str, ...]) -> pd.DataFrame:
    """Slim flagged frame (filtered) used by the matchup/dismissal section."""
    d = load_flagged()
    if years:
        d = d[d["year"].isin(years)]
    if countries:
        d = d[d["bowling_team"].isin(countries)]
    keep = ["bowler", "batting_team", "wicket_kind", "bowler_wicket",
            "is_legal", "is_dot", "is_boundary", "runs_off_bowler",
            "runs_batter", "extra_wides", "extra_noballs"]
    return d[[c for c in keep if c in d.columns]].copy()


@st.cache_data(show_spinner="Building development trajectories…")
def build_trajectory_frame(years: tuple[int, ...],
                           countries: tuple[str, ...]) -> pd.DataFrame:
    """Slim, chronologically-aware flagged frame for the trajectory view."""
    d = load_flagged()
    if years:
        d = d[d["year"].isin(years)]
    if countries:
        d = d[d["bowling_team"].isin(countries)]
    keep = ["bowler", "match_id", "match_date", "is_legal", "is_dot",
            "is_boundary", "bowler_wicket", "runs_off_bowler"]
    return d[[c for c in keep if c in d.columns]].copy()


@st.cache_data(show_spinner=False)
def trajectory_match_counts(years: tuple[int, ...],
                            countries: tuple[str, ...]) -> pd.Series:
    """Matches in which each bowler sent down at least one legal ball."""
    frame = build_trajectory_frame(years, countries)
    legal = frame[frame["is_legal"].astype(bool)]
    return legal.groupby("bowler")["match_id"].nunique()


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def stable_bowler_selectbox(label: str, options: list[str], key: str) -> str | None:
    """Selectbox that keeps the user's pick across filter changes.

    Streamlit would raise / silently reset if the stored selection is no longer
    a valid option, so we only drop it when the bowler has left the cohort —
    otherwise the same bowler stays selected when filters change."""
    if not options:
        return None
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)  # bowler left the cohort -> reset to first
    return st.selectbox(label, options, key=key)


# --------------------------------------------------------------------------- #
# Page config & global font
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Bowl Metrics", layout="wide")

# Georgia everywhere. We override Streamlit's own font CSS variables and set
# the family on every element (universal selector), so tab labels, widget text,
# tables and the body all use it regardless of Streamlit's internal class names.
GEORGIA = "Georgia, 'Times New Roman', serif"
st.markdown(
    f"""
    <style>
    :root {{
        --font: {GEORGIA};
        --font-family: {GEORGIA};
    }}
    /* Apply Georgia to text elements only — never use * so icon fonts survive */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    p, h1, h2, h3, h4, h5, h6, li, a, td, th,
    label, button, input, textarea, select,
    .stMarkdown, [data-testid="stText"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stExpander"] summary p,
    [data-testid="stWidgetLabel"] {{
        font-family: {GEORGIA} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Filters live inside each tab (not the sidebar) so they sit next to the chart
# they drive.
all_years, all_countries = filter_options()


# --------------------------------------------------------------------------- #
# Performance Fingerprint
# --------------------------------------------------------------------------- #
def render_fingerprint(sel_years, sel_countries, bowler, min_balls):
  st.subheader("Performance Fingerprint")

  # Phase is a per-view display control (it changes which cohort the percentiles
  # are measured within), so it stays in the tab rather than the shared sidebar.
  st.session_state.setdefault("fp_phase", "All overs")
  phase_choice = st.selectbox(
      "Phase", ["All overs"] + list(PHASES.keys()), key="fp_phase",
      help="Percentiles are computed within the selected phase.",
  )
  phase = None if phase_choice == "All overs" else phase_choice

  # Phase-specific table drives the actual fingerprint (peers within the phase).
  table = build_table(phase, min_balls, tuple(sel_years), tuple(sel_countries))
  cohort = table[table["in_cohort"]].copy()

  # The selected bowler may not have bowled enough in this particular phase.
  if bowler not in cohort.index:
    balls_here = int(table.loc[bowler, "balls_legal"]) if bowler in table.index else 0
    st.info(
      f"{bowler} bowled only {balls_here} legal ball(s) in the "
      f"{phase_choice.lower()} under these filters — not enough for a fingerprint "
      f"at the current min-balls setting ({min_balls}). Lower the slider, switch "
      f"to a phase he bowls more in, or keep him selected and choose another phase."
    )
    return

  yr_txt = ", ".join(map(str, sel_years)) if sel_years else "all years"
  ctry_txt = ", ".join(sel_countries) if sel_countries else "all teams"

  st.caption(
    f"{bowler} — {phase_choice.lower()} · {yr_txt} · {ctry_txt} · "
    f"cohort = {len(cohort)} bowlers with ≥ {min_balls} legal balls"
  )

  row = table.loc[bowler]

  k = st.columns(4)
  kpis = [
    ("Economy", f"{row['economy']:.2f}", "rpo"),
    ("Strike rate", f"{row['strike_rate']:.1f}" if pd.notna(row['strike_rate']) else "—", "balls/wkt"),
    ("Dot-ball %", f"{row['dot_pct']:.1f}%", ""),
    ("Boundary %", f"{row['boundary_pct']:.1f}%", ""),
  ]
  for col, (lbl, val, unit) in zip(k, kpis):
    col.metric(lbl, val, unit)

  st.divider()

  metric_keys = list(FINGERPRINT_METRICS)
  labels = [FINGERPRINT_METRICS[m]["label"] for m in metric_keys]
  pctiles = [row.get(f"{m}_pctile") for m in metric_keys]
  medians = [cohort[m].median() for m in metric_keys]
  better_low = [FINGERPRINT_METRICS[m]["lower_is_better"] for m in metric_keys]

  def _raw_disp(m: str, v) -> str:
    if pd.isna(v):
        return "—"
    if m == "economy":
        return f"{v:.2f}"
    if m == "strike_rate":
        return f"{v:.1f}"
    return f"{v:.1f}%"

  raw_disp = [_raw_disp(m, row[m]) for m in metric_keys]
  med_disp = [_raw_disp(m, md) for m, md in zip(metric_keys, medians)]
  # axis labels carry the real stat so the chart can't be read backwards
  axis_labels = [f"{lab}<br>{rd}" for lab, rd in zip(labels, raw_disp)]
  green, red, grey = "#27AE60", "#E74C3C", "#9aa0a6"

  left, right = st.columns([1, 1])

  with left:
    st.subheader("Fingerprint vs peers")
    fig = go.Figure()
    # grey reference ring = cohort median (50th percentile on every axis)
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(metric_keys) + [50],
        theta=axis_labels + [axis_labels[0]],
        mode="lines", line=dict(color=grey, dash="dash"),
        name="Peer median", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=pctiles + [pctiles[0]],
        theta=axis_labels + [axis_labels[0]],
        fill="toself", name=bowler, line_color="#2E86DE",
        customdata=[[rd, md] for rd, md in zip(raw_disp, med_disp)]
                   + [[raw_disp[0], med_disp[0]]],
        hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.0f}th"
                      "<br>Value: %{customdata[0]}"
                      "<br>Peer median: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(
        font=dict(family="Georgia, serif"),
        polar=dict(
            domain=dict(x=[0.12, 0.88], y=[0.10, 0.90]),  # shrink plot, leave room for labels
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75],
                            tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True, legend=dict(orientation="h", y=-0.12),
        height=480, margin=dict(l=90, r=90, t=50, b=70),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distance from centre = percentile rank (0–100). Outside the grey "
               "ring = better than the typical peer; inside = worse. Each axis "
               "shows the bowler's actual figure.")

  with right:
    st.subheader("Better or worse than a typical peer")
    diffs = [(p - 50) if pd.notna(p) else 0 for p in pctiles]  # +ve = better
    colors = [green if d >= 0 else red for d in diffs]
    # label shows the real stat and where it sits vs the median
    bar_text = [
        f"{rd}  ({p:.0f}th)" if pd.notna(p) else "—"
        for rd, p in zip(raw_disp, pctiles)
    ]
    bar = go.Figure(go.Bar(
        x=diffs, y=labels, orientation="h",
        marker_color=colors, text=bar_text, textposition="outside",
        customdata=[[rd, md, "lower" if lo else "higher"]
                    for rd, md, lo in zip(raw_disp, med_disp, better_low)],
        hovertemplate="<b>%{y}</b><br>Value: %{customdata[0]}"
                      "<br>Peer median: %{customdata[1]}"
                      "<br>Better when %{customdata[2]}<extra></extra>",
    ))
    bar.add_vline(x=0, line_color=grey)
    bar.add_annotation(x=42, y=-0.8, text="better →", showarrow=False,
                       font=dict(color=green, size=12))
    bar.add_annotation(x=-42, y=-0.8, text="← worse", showarrow=False,
                       font=dict(color=red, size=12))
    bar.update_layout(
        font=dict(family="Georgia, serif"),
        xaxis=dict(range=[-55, 55],
                   tickvals=[-50, -25, 0, 25, 50],
                   ticktext=["worse", "", "median", "", "better"],
                   title="Percentile gap from the peer median"),
        yaxis=dict(autorange="reversed"),
        height=480, margin=dict(l=10, r=10, t=30, b=40), showlegend=False,
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption("Bars point right/green when this bowler beats the peer median on "
               "that metric, left/red when worse. The number on each bar is the "
               "actual stat, with its percentile in brackets.")

  st.divider()
  st.subheader("Detail")

  detail = pd.DataFrame({
    "Metric": labels,
    "Raw value": [row[m] for m in metric_keys],
    "Percentile": pctiles,
    "Cohort median": [cohort[m].median() for m in metric_keys],
    "Better when": ["Lower" if FINGERPRINT_METRICS[m]["lower_is_better"] else "Higher"
                    for m in metric_keys],
  })
  st.dataframe(
    detail.style.format({
        "Raw value": "{:.2f}", "Percentile": "{:.0f}", "Cohort median": "{:.2f}",
    }),
    use_container_width=True, hide_index=True,
  )

  with st.expander("Full cohort leaderboard"):
    show = cohort[["overs", "wickets", "economy", "strike_rate",
                   "dot_pct", "boundary_pct",
                   "economy_pctile", "strike_rate_pctile",
                   "dot_pct_pctile", "boundary_pct_pctile"]]
    st.dataframe(show.sort_values("economy"), use_container_width=True)


# --------------------------------------------------------------------------- #
# Role Identification
# --------------------------------------------------------------------------- #
def render_roles(sel_years, sel_countries, bowler, min_balls):
  st.subheader("Role Identification")
  st.caption(
    "k-means learns three bowling archetypes from how well each bowler performs "
    "in every phase, then checks whether he is actually *bowled* in the phase "
    "his skill profile suits."
  )

  roles = build_roles(min_balls, 30, tuple(sel_years), tuple(sel_countries))
  if roles is None or roles.empty:
    st.warning("Not enough bowlers to cluster with these filters. Widen the "
               "year/country filter or lower the role min-balls threshold in the "
               "sidebar.")
    return

  if bowler not in roles.index:
    st.info(f"{bowler} doesn't clear the role-cohort minimum ({min_balls} legal "
            f"balls overall) under these filters, so he wasn't clustered. Lower "
            f"the role min-balls threshold in the sidebar to include him.")
    return

  row = roles.loc[bowler]
  archetype = row["archetype"]
  arch_phase = row["archetype_phase"]
  dep_phase = row["deployment_phase"]
  dep_share = row[f"{dep_phase}_share"]

  # --- This bowler's phase profile (skill + workload per phase) -------------
  st.subheader(f"{bowler}'s phase profile")
  cards = st.columns(len(ROLE_PHASE_ORDER))
  for col, phase in zip(cards, ROLE_PHASE_ORDER):
    label = f"{phase} ★ role" if phase == arch_phase else phase
    col.metric(
        label, f"{row[phase]:.0f}/100",
        f"{row[f'{phase}_share']:.0f}% of his balls", delta_color="off",
    )
  st.caption("★ marks his archetype phase — where his skill profile is strongest. "
             "Each card shows his 0–100 phase skill and the share of his balls "
             "bowled in that phase.")

  st.divider()

  # --- Per-bowler verdict ---------------------------------------------------
  if row["in_role"]:
    st.success(
      f"✓ **In role** — {bowler} profiles as a **{archetype}**, and is bowled "
      f"mostly in the **{dep_phase}** ({dep_share:.0f}% of his balls). His "
      f"deployment matches his strongest phase."
    )
  else:
    st.warning(
      f"⚠ **Possible mismatch** — {bowler} profiles as a **{archetype}** "
      f"(strongest in the **{arch_phase}**), but is bowled mostly in the "
      f"**{dep_phase}** ({dep_share:.0f}% of his balls). He may be under-used "
      f"in the phase he is best at."
    )

  # --- Skill vs workload ----------------------------------------------------
  left, right = st.columns([1, 1])

  with left:
    st.subheader("Where he's good vs where he bowls")
    skill_vals = [row[p] for p in ROLE_PHASE_ORDER]
    share_vals = [row[f"{p}_share"] for p in ROLE_PHASE_ORDER]
    cmp = go.Figure()
    cmp.add_trace(go.Bar(
        x=ROLE_PHASE_ORDER, y=skill_vals, name="Phase skill (0–100)",
        marker_color="#2E86DE",
        hovertemplate="<b>%{x}</b><br>Skill: %{y:.0f}/100<extra></extra>",
    ))
    cmp.add_trace(go.Bar(
        x=ROLE_PHASE_ORDER, y=share_vals, name="Workload share (%)",
        marker_color="#9aa0a6",
        hovertemplate="<b>%{x}</b><br>Workload: %{y:.0f}%<extra></extra>",
    ))
    cmp.update_layout(
        font=dict(family="Georgia, serif"),
        barmode="group", height=430,
        yaxis=dict(title="Skill score / workload %", range=[0, 100]),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=30, b=40),
    )
    st.plotly_chart(cmp, use_container_width=True)
    st.caption("Blue = how good he is in each phase (percentile skill). Grey = "
               "how much he actually bowls there. A tall blue bar with a short "
               "grey bar beside it is the mismatch.")

  with right:
    st.subheader("Cluster map")
    fig = go.Figure()
    for phase in ROLE_PHASE_ORDER:
      grp = roles[roles["archetype"] == ROLE_ARCHETYPE[phase]]
      fig.add_trace(go.Scatterternary(
          a=grp["Powerplay"], b=grp["Middle"], c=grp["Death"],
          mode="markers", name=ROLE_ARCHETYPE[phase],
          marker=dict(size=7, color=PHASE_COLORS[phase], opacity=0.55),
          text=grp.index,
          hovertemplate="<b>%{text}</b><br>PP %{a:.0f} · Mid %{b:.0f} · "
                        "Death %{c:.0f}<extra></extra>",
      ))
    fig.add_trace(go.Scatterternary(
        a=[row["Powerplay"]], b=[row["Middle"]], c=[row["Death"]],
        mode="markers", name=bowler,
        marker=dict(size=18, symbol="star", color="#C0392B",
                    line=dict(width=1, color="white")),
        hovertemplate=f"<b>{bowler}</b><br>PP %{{a:.0f}} · Mid %{{b:.0f}} · "
                      "Death %{c:.0f}<extra></extra>",
    ))
    fig.update_layout(
        font=dict(family="Georgia, serif"),
        ternary=dict(
            sum=100,
            aaxis=dict(title="Powerplay", color=PHASE_COLORS["Powerplay"]),
            baxis=dict(title="Middle", color=PHASE_COLORS["Middle"]),
            caxis=dict(title="Death", color=PHASE_COLORS["Death"]),
        ),
        height=430, legend=dict(orientation="h", y=-0.08),
        margin=dict(l=40, r=40, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each point is a bowler placed by his phase-skill mix; colour is "
               "the archetype his cluster maps to. The red star is the selected "
               "bowler — corners pull toward the phase he's strongest in.")

  st.divider()
  st.subheader("Detail")

  detail = pd.DataFrame({
    "Phase": ROLE_PHASE_ORDER,
    "Skill (0–100)": [row[p] for p in ROLE_PHASE_ORDER],
    "Workload share %": [row[f"{p}_share"] for p in ROLE_PHASE_ORDER],
  })
  st.dataframe(
    detail.style.format({"Skill (0–100)": "{:.0f}", "Workload share %": "{:.0f}"}),
    use_container_width=True, hide_index=True,
  )
  st.caption(
    f"Archetype: **{archetype}** · profiled-strongest phase: **{arch_phase}** · "
    f"most-bowled phase: **{dep_phase}** · "
    f"verdict: {'in role' if row['in_role'] else 'mismatch'}."
  )

  with st.expander("All bowlers — archetype & deployment"):
    show = roles[["archetype", "Powerplay", "Middle", "Death",
                  "Powerplay_share", "Middle_share", "Death_share",
                  "deployment_phase", "in_role", "total_balls"]].copy()
    show = show.rename(columns={
        "Powerplay": "PP skill", "Middle": "Mid skill", "Death": "Death skill",
        "Powerplay_share": "PP %", "Middle_share": "Mid %", "Death_share": "Death %",
    })
    st.dataframe(
        show.sort_values("total_balls", ascending=False).style.format({
            "PP skill": "{:.0f}", "Mid skill": "{:.0f}", "Death skill": "{:.0f}",
            "PP %": "{:.0f}", "Mid %": "{:.0f}", "Death %": "{:.0f}",
            "total_balls": "{:.0f}",
        }),
        use_container_width=True,
    )

  with st.expander("How the archetypes were learned"):
    st.dataframe(
        cluster_summary(roles).style.format({
            "Powerplay": "{:.0f}", "Middle": "{:.0f}", "Death": "{:.0f}",
            "pct_in_role": "{:.0f}",
        }),
        use_container_width=True,
    )
    st.caption("Mean phase-skill of each learned cluster (it peaks on the phase "
               "it is named after) and the share of its bowlers actually "
               "deployed in that phase.")


# --------------------------------------------------------------------------- #
# Execution Consistency & Matchup
# --------------------------------------------------------------------------- #
def render_execution(sel_years, sel_countries, bowler, min_overs):
  st.subheader("Execution & Matchups")
  st.caption(
    "Two coaching lenses: how *repeatable* a bowler's overs are (a concrete "
    "drill target), and *how* he takes wickets and *where* his runs leak."
  )

  cons, ot = build_consistency(min_overs, tuple(sel_years), tuple(sel_countries))
  cohort = cons[cons["in_cohort"]].copy()
  if cohort.empty:
    st.warning("No bowlers reach the minimum-overs threshold with these filters. "
               "Widen the year/country filter or lower the min-overs slider in "
               "the sidebar.")
    return

  if bowler not in cohort.index:
    bowled = int(cons.loc[bowler, "overs_bowled"]) if bowler in cons.index else 0
    st.info(f"{bowler} has only {bowled} over(s) under these filters — below the "
            f"{min_overs}-over cohort minimum, so his consistency percentile "
            f"isn't measured. Lower the min-overs threshold in the sidebar.")
    return

  row = cons.loc[bowler]

  yr_txt = ", ".join(map(str, sel_years)) if sel_years else "all years"
  ctry_txt = ", ".join(sel_countries) if sel_countries else "all teams"
  st.caption(
    f"{bowler} · {yr_txt} · {ctry_txt} · "
    f"cohort = {len(cohort)} bowlers with ≥ {min_overs} overs"
  )

  # ======================================================================= #
  # SECTION 1 — Execution consistency
  # ======================================================================= #
  st.header("Execution consistency")
  st.caption(
    "Each over he bowls is one observation of runs conceded. The *spread* of "
    "that distribution is how repeatable he is — tighter spread = a more "
    "coachable, reproducible groove."
  )

  bowler_overs = ot[ot["bowler"] == bowler]["over_runs"]
  std_med = cohort["std_runs"].median()
  best_std = cohort["std_runs"].min()
  pctile = row["consistency_pctile"]

  k = st.columns(4)
  k[0].metric("Mean runs / over", f"{row['mean_runs']:.2f}")
  k[1].metric("Spread (std dev)", f"{row['std_runs']:.2f}",
              f"{row['std_runs'] - std_med:+.2f} vs peer median",
              delta_color="inverse")
  k[2].metric("Consistency percentile",
              f"{pctile:.0f}th" if pd.notna(pctile) else "—")
  k[3].metric("Overs bowled", f"{int(row['overs_bowled'])}")

  st.divider()
  green, red, grey, blue = "#27AE60", "#E74C3C", "#9aa0a6", "#2E86DE"
  left, right = st.columns([1, 1])

  with left:
    st.subheader("How his overs spread out")
    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=bowler_overs, xbins=dict(start=-0.5, size=1),
        marker_color=blue, name="Overs",
        hovertemplate="%{x} runs in the over<br>%{y} overs<extra></extra>",
    ))
    hist.add_vline(x=row["mean_runs"], line_color=red, line_dash="dash",
                   annotation_text=f"mean {row['mean_runs']:.1f}",
                   annotation_position="top")
    hist.update_layout(
        font=dict(family="Georgia, serif"),
        xaxis=dict(title="Runs conceded in an over"),
        yaxis=dict(title="Number of overs"),
        height=430, margin=dict(l=10, r=10, t=30, b=40), showlegend=False,
    )
    st.plotly_chart(hist, use_container_width=True)
    st.caption("A tall, narrow pile near the mean is a repeatable bowler. A wide, "
               "flat spread — cheap overs mixed with expensive ones — is the "
               "execution problem to drill out.")

  with right:
    st.subheader("Shape of his overs")
    buckets = over_bucket_shares(bowler_overs)
    bcolors = [green, blue, "#E67E22", red]
    bar = go.Figure(go.Bar(
        x=buckets["pct"], y=buckets["band"], orientation="h",
        marker_color=bcolors,
        text=[f"{p:.0f}%  ({n})" for p, n in zip(buckets["pct"], buckets["overs"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.0f}% of his overs<extra></extra>",
    ))
    bar.update_layout(
        font=dict(family="Georgia, serif"),
        xaxis=dict(title="% of his overs", range=[0, max(60, buckets['pct'].max() + 12)]),
        yaxis=dict(autorange="reversed"),
        height=430, margin=dict(l=10, r=10, t=30, b=40), showlegend=False,
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption("The share of overs landing in each runs band. Shrinking the "
               "'Costly' and 'Leaked' slices is the measurable goal.")

  # --- Drill-target verdict -------------------------------------------------
  if pd.notna(pctile) and pd.notna(best_std):
    gap = row["std_runs"] - best_std
    if pctile >= 67:
      st.success(
        f"✓ **Repeatable** — {bowler} is in the top third of the cohort for "
        f"over-to-over consistency ({pctile:.0f}th percentile, spread "
        f"{row['std_runs']:.2f}). His execution is already a strength to protect."
      )
    elif pctile >= 34:
      st.info(
        f"**Mid-pack consistency** — {pctile:.0f}th percentile (spread "
        f"{row['std_runs']:.2f}). Closing the {gap:.2f}-run gap to the cohort's "
        f"tightest bowler (spread {best_std:.2f}) would move him toward elite "
        f"repeatability — a concrete drill target."
      )
    else:
      st.warning(
        f"⚠ **Inconsistent** — {pctile:.0f}th percentile (spread "
        f"{row['std_runs']:.2f} vs the cohort's tightest {best_std:.2f}). His "
        f"overs swing widely; tightening this {gap:.2f}-run spread is the single "
        f"clearest execution drill target."
      )

  st.divider()

  # ======================================================================= #
  # SECTION 2 — Matchup & dismissal profile
  # ======================================================================= #
  st.header("Matchup & dismissal profile")
  st.caption(
    "How he takes his wickets, where his runs leak, and which opposition he "
    "dominates or struggles against."
  )

  prof = build_profile_frame(tuple(sel_years), tuple(sel_countries))

  d_left, d_right = st.columns([1, 1])

  with d_left:
    st.subheader("How he takes wickets")
    dp = dismissal_profile(prof, bowler)
    if dp.empty:
      st.info(f"{bowler} has no bowler-credited wickets under these filters.")
    else:
      coh = dismissal_profile(prof)  # whole cohort for a comparison line
      coh_share = (coh / coh.sum() * 100).reindex(dp.index).fillna(0)
      his_share = (dp / dp.sum() * 100)
      ddf = go.Figure()
      ddf.add_trace(go.Bar(
          x=dp.index.str.title(), y=his_share.values, name="His %",
          marker_color=blue,
          text=[f"{v:.0f}%" for v in his_share.values], textposition="outside",
          customdata=dp.values,
          hovertemplate="<b>%{x}</b><br>%{y:.0f}% of his wickets"
                        "<br>(%{customdata} dismissals)<extra></extra>",
      ))
      ddf.add_trace(go.Scatter(
          x=dp.index.str.title(), y=coh_share.values, name="Cohort %",
          mode="markers", marker=dict(color=grey, size=11, symbol="diamond"),
          hovertemplate="<b>%{x}</b><br>Cohort: %{y:.0f}%<extra></extra>",
      ))
      ddf.update_layout(
          font=dict(family="Georgia, serif"),
          yaxis=dict(title="% of wickets",
                     range=[0, max(his_share.max(), coh_share.max()) + 15]),
          legend=dict(orientation="h", y=-0.2),
          height=420, margin=dict(l=10, r=10, t=30, b=40),
      )
      st.plotly_chart(ddf, use_container_width=True)
      st.caption(f"Bars = {bowler}'s wicket mix; grey diamonds = the cohort's. "
                 "Bars above the diamond are his signature modes of dismissal.")

  with d_right:
    st.subheader("Where his runs leak")
    leak = runs_leak_breakdown(prof, bowler)
    pie = go.Figure(go.Pie(
        labels=leak["source"], values=leak["runs"], hole=0.45,
        marker=dict(colors=[red, blue, "#E67E22"]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} runs (%{percent})<extra></extra>",
    ))
    pie.update_layout(
        font=dict(family="Georgia, serif"),
        showlegend=False, height=420, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(pie, use_container_width=True)
    bdry_pct = leak.loc[leak["source"].str.startswith("Boundaries"), "pct"].iloc[0]
    st.caption(f"{bdry_pct:.0f}% of his runs are conceded through the rope. A high "
               "boundary share points to length/variation work; a high extras "
               "slice points to a discipline drill.")

  st.divider()
  st.subheader("Opposition matchups")
  mu = opposition_matchup(prof, bowler, min_balls=24)
  if mu.empty:
    st.info("Not enough balls against any single opposition (need ≥ 24 legal "
            "balls vs a team) to show matchups under these filters.")
  else:
    best = mu.head(1).index[0]
    worst = mu.tail(1).index[0]
    c1, c2 = st.columns(2)
    c1.metric("Best matchup (lowest economy)", best,
              f"{mu.loc[best, 'economy']:.2f} rpo")
    c2.metric("Toughest matchup (highest economy)", worst,
              f"{mu.loc[worst, 'economy']:.2f} rpo", delta_color="inverse")

    show = mu.rename_axis("Opposition").reset_index()
    show = show[["Opposition", "overs", "economy", "wickets", "boundaries"]]
    mbar = go.Figure(go.Bar(
        x=show["economy"], y=show["Opposition"], orientation="h",
        marker_color=[green if e <= mu["economy"].median() else red
                      for e in show["economy"]],
        text=[f"{e:.2f}" for e in show["economy"]], textposition="outside",
        customdata=show[["overs", "wickets"]].values,
        hovertemplate="<b>%{y}</b><br>Economy %{x:.2f} rpo"
                      "<br>%{customdata[0]} overs · %{customdata[1]} wkts<extra></extra>",
    ))
    mbar.update_layout(
        font=dict(family="Georgia, serif"),
        xaxis=dict(title="Economy (rpo) by opposition"),
        yaxis=dict(autorange="reversed"),
        height=max(300, 60 + 26 * len(show)),
        margin=dict(l=10, r=10, t=30, b=40), showlegend=False,
    )
    st.plotly_chart(mbar, use_container_width=True)
    st.caption("Green = he keeps that team below his median economy; red = they "
               "get after him. Surfaces who to bowl him against and who to hide "
               "him from.")

    with st.expander("Matchup table — full breakdown"):
      st.dataframe(
          show.style.format({"overs": "{:.1f}", "economy": "{:.2f}",
                             "wickets": "{:.0f}", "boundaries": "{:.0f}"}),
          use_container_width=True, hide_index=True,
      )

  with st.expander("Most consistent bowlers in the cohort"):
    lb = cohort[["overs_bowled", "mean_runs", "std_runs",
                 "consistency_pctile"]].copy()
    lb = lb.rename(columns={
        "overs_bowled": "Overs", "mean_runs": "Mean r/o",
        "std_runs": "Spread (std)", "consistency_pctile": "Consistency pctile",
    })
    st.dataframe(
        lb.sort_values("Spread (std)").style.format({
            "Overs": "{:.0f}", "Mean r/o": "{:.2f}",
            "Spread (std)": "{:.2f}", "Consistency pctile": "{:.0f}",
        }),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Development trajectory
# --------------------------------------------------------------------------- #
def render_trajectory(sel_years, sel_countries, bowler, window, min_matches):
  st.subheader("Development Trajectory")
  st.caption(
    "Trend and change-point detection on a bowler's rolling metrics — so you "
    "can tell whether the things you've coached are actually showing up over "
    "time, or whether the recent numbers are just noise."
  )

  st.session_state.setdefault("traj_metric", TRAJECTORY_METRICS["economy"]["label"])

  # Focus-metric selector is a per-view display control, so it stays in the tab.
  label_to_key = {m["label"]: k for k, m in TRAJECTORY_METRICS.items()}
  focus_label = st.selectbox(
      "Focus metric", list(label_to_key), key="traj_metric",
      help="The metric shown in the big trajectory chart below. The summary "
           "table covers all four.",
  )
  metric = label_to_key[focus_label]
  meta = TRAJECTORY_METRICS[metric]
  fmt = meta["fmt"]; lower_better = meta["lower_is_better"]

  frame = build_trajectory_frame(tuple(sel_years), tuple(sel_countries))
  tl = bowler_match_timeline(frame, bowler)
  if len(tl) < max(window + 2, min_matches):
    st.info(f"{bowler} has only {len(tl)} match(es) under these filters — not "
            f"enough for a stable {window}-match rolling trajectory. Lower the "
            f"window or widen the filters.")
    return

  yr_txt = ", ".join(map(str, sel_years)) if sel_years else "all years"
  ctry_txt = ", ".join(sel_countries) if sel_countries else "all teams"
  st.caption(
    f"{bowler} · {yr_txt} · {ctry_txt} · {len(tl)} matches · "
    f"{window}-match rolling window"
  )

  # --- Trend + change-points for the focus metric ---------------------------
  roll = rolling_metric(tl, metric, window)
  roll_valid = roll.dropna()
  trend = linear_trend(roll_valid)

  cf = changepoint_frame(tl, metric)
  cps = detect_change_points(cf[metric].to_numpy(), cf["balls"].to_numpy())
  segs = segment_levels(cf, metric, cps) if len(cf) >= 6 else pd.DataFrame()

  career = pooled_metric(tl, metric)
  latest = float(roll_valid.iloc[-1]) if not roll_valid.empty else np.nan
  green, red, grey, blue = "#27AE60", "#E74C3C", "#9aa0a6", "#2E86DE"

  def _improve_color(delta_good: bool) -> str:
    return green if delta_good else red

  # --- KPI row --------------------------------------------------------------
  k = st.columns(4)
  k[0].metric(f"Career {meta['label'].lower()}", fmt.format(career))
  if np.isfinite(latest) and np.isfinite(career):
    moved = latest - career
    better = (moved < 0) if lower_better else (moved > 0)
    k[1].metric(f"Latest {window}-match", fmt.format(latest),
                f"{moved:+.2f} vs career",
                delta_color="normal" if better else "inverse")
  else:
    k[1].metric(f"Latest {window}-match", "—")

  if trend is not None and np.isfinite(trend["p"]):
    improving = (trend["slope"] < 0) if lower_better else (trend["slope"] > 0)
    if trend["p"] < 0.05:
      verdict = "Improving" if improving else "Worsening"
    else:
      verdict = "Flat"
    k[2].metric("Trend", verdict,
                f"{trend['slope']*10:+.2f} / 10 matches",
                delta_color=("normal" if verdict == "Improving"
                             else "inverse" if verdict == "Worsening" else "off"))
  else:
    k[2].metric("Trend", "—")
  k[3].metric("Change-points found", f"{len(cps)}")

  st.divider()
  left, right = st.columns([1.25, 1])

  # --- Main trajectory chart (focus metric) ---------------------------------
  with left:
    st.subheader(f"{meta['label']} over time")
    fig = go.Figure()
    # faint per-match underlay
    fig.add_trace(go.Scatter(
        x=tl["match_date"], y=tl[metric], mode="markers",
        marker=dict(color=grey, size=5, opacity=0.45), name="Per match",
        hovertemplate="%{x|%d %b %Y}<br>" + meta["label"] + ": %{y:.2f}<extra></extra>",
    ))
    # rolling line
    fig.add_trace(go.Scatter(
        x=tl["match_date"], y=roll, mode="lines", line=dict(color=blue, width=3),
        name=f"{window}-match rolling",
        hovertemplate="%{x|%d %b %Y}<br>Rolling: %{y:.2f}<extra></extra>",
    ))
    # OLS trend over the rolling line
    if trend is not None and not roll_valid.empty:
      x_tr = tl["match_date"].iloc[-len(roll_valid):]
      y_tr = trend["intercept"] + trend["slope"] * np.arange(len(roll_valid))
      fig.add_trace(go.Scatter(
          x=x_tr, y=y_tr, mode="lines",
          line=dict(color="#8E44AD", width=2, dash="dot"), name="Trend",
          hovertemplate="Trend: %{y:.2f}<extra></extra>",
      ))
    # change-point step levels + markers
    if not segs.empty:
      for _, sg in segs.iterrows():
        fig.add_trace(go.Scatter(
            x=[sg["start_date"], sg["end_date"]], y=[sg["level"], sg["level"]],
            mode="lines", line=dict(color="#E67E22", width=2), showlegend=False,
            hovertemplate=f"Segment level: {sg['level']:.2f}<extra></extra>",
        ))
      for _, sg in segs.iloc[1:].iterrows():
        cp_date = sg["start_date"]
        if pd.notna(cp_date):
          fig.add_vline(x=cp_date, line_color="#E67E22", line_dash="dash",
                        opacity=0.6)
    fig.update_layout(
        font=dict(family="Georgia, serif"),
        yaxis=dict(title=f"{meta['label']} ({meta['unit']})"),
        xaxis=dict(title="Match date"),
        height=470, legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey dots are single matches (noisy). The blue line is the "
               "rolling rate; the dotted purple line its trend. Orange steps "
               "are the detected level changes — the match around which the "
               "metric genuinely shifted.")

  # --- Improvement index across all four metrics ----------------------------
  with right:
    st.subheader("Improvement index — all metrics")
    idx_fig = go.Figure()
    palette = {"economy": blue, "dot_pct": green,
               "boundary_pct": "#E67E22", "strike_rate": "#8E44AD"}
    plotted = False
    for m, mmeta in TRAJECTORY_METRICS.items():
      r = rolling_metric(tl, m, window)
      rv = r.dropna()
      if rv.empty:
        continue
      base = rv.iloc[0]
      if not np.isfinite(base) or base == 0:
        continue
      # Orient so up = better regardless of the metric's direction.
      idx = (base / r) * 100 if mmeta["lower_is_better"] else (r / base) * 100
      idx_fig.add_trace(go.Scatter(
          x=tl["match_date"], y=idx, mode="lines",
          line=dict(color=palette[m], width=2), name=mmeta["label"],
          hovertemplate=mmeta["label"] + ": %{y:.0f}<extra></extra>",
      ))
      plotted = True
    idx_fig.add_hline(y=100, line_color=grey, line_dash="dash")
    idx_fig.update_layout(
        font=dict(family="Georgia, serif"),
        yaxis=dict(title="Improvement index (start = 100)"),
        xaxis=dict(title="Match date"),
        height=470, legend=dict(orientation="h", y=-0.22),
        margin=dict(l=10, r=10, t=30, b=40),
    )
    if plotted:
      st.plotly_chart(idx_fig, use_container_width=True)
      st.caption("Each metric's rolling value rebased to 100 at the start and "
                 "flipped so up always means better. Lines climbing above the "
                 "grey 100-line are the areas where development is showing up.")
    else:
      st.info("Not enough rolling history to index the metrics.")

  # --- Verdict on the focus metric ------------------------------------------
  if not segs.empty and len(segs) >= 2:
    before = segs["level"].iloc[-2]
    after = segs["level"].iloc[-1]
    cp_date = segs["start_date"].iloc[-1]
    when = cp_date.strftime("%b %Y") if pd.notna(cp_date) else "recently"
    step_p = _welch_p(cf[metric].iloc[:cps[-1]], cf[metric].iloc[cps[-1]:])
    moved = after - before
    better = (moved < 0) if lower_better else (moved > 0)
    sig = (np.isfinite(step_p) and step_p < 0.05)
    direction = "down" if moved < 0 else "up"
    good = "the coached direction" if better else "the wrong direction"
    p_txt = f"p={step_p:.3f}" if np.isfinite(step_p) else "p n/a"
    if better and sig:
      st.success(
        f"✓ **Improvement is showing up** — around **{when}**, {bowler}'s "
        f"{meta['label'].lower()} stepped {direction} from "
        f"{fmt.format(before)} to {fmt.format(after)} ({meta['unit']}), a "
        f"significant shift in {good} ({p_txt})."
      )
    elif (not better) and sig:
      st.warning(
        f"⚠ **Going the wrong way** — around **{when}**, {bowler}'s "
        f"{meta['label'].lower()} stepped {direction} from "
        f"{fmt.format(before)} to {fmt.format(after)} ({meta['unit']}), a "
        f"significant move in {good} ({p_txt})."
      )
    else:
      st.info(
        f"**Step detected but not conclusive** — around **{when}**, "
        f"{meta['label'].lower()} moved from {fmt.format(before)} to "
        f"{fmt.format(after)} ({meta['unit']}), but the change isn't "
        f"statistically clear ({p_txt}). Keep gathering matches before calling "
        f"it a real shift."
      )
  elif trend is not None and np.isfinite(trend["p"]) and trend["p"] < 0.05:
    improving = (trend["slope"] < 0) if lower_better else (trend["slope"] > 0)
    word = "improving" if improving else "worsening"
    st.info(
      f"No single step stands out, but the {window}-match trend is **{word}** "
      f"steadily ({trend['slope']*10:+.2f} {meta['unit']} per 10 matches, "
      f"p={trend['p']:.3f}) — a gradual drift rather than a sudden change."
    )
  else:
    st.info(f"{bowler}'s {meta['label'].lower()} shows no significant trend or "
            f"step under these filters — it's been broadly stable.")

  st.divider()
  st.subheader("All metrics — trend & latest step")
  summ = trajectory_summary(tl, window)
  show = pd.DataFrame({
    "Metric": summ["label"],
    "Career": summ["career"],
    f"Latest ({window}-match)": summ["latest"],
    "Slope /10 matches": summ["slope_per10"],
    "Trend": summ["direction"],
    "Step before": summ["step_before"],
    "Step after": summ["step_after"],
    "Step p": summ["step_p"],
  })
  st.dataframe(
    show.style.format({
        "Career": "{:.2f}", f"Latest ({window}-match)": "{:.2f}",
        "Slope /10 matches": "{:+.2f}", "Step before": "{:.2f}",
        "Step after": "{:.2f}", "Step p": "{:.3f}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
  )
  st.caption("Trend reads each metric in its improvement direction: 'Improving' "
             "= moving the good way with a significant slope (p<0.05), 'Flat' = "
             "no clear slope. 'Step' columns are the most recent detected level "
             "change and its significance.")

  with st.expander("Match-by-match timeline"):
    tdf = tl[["seq", "match_date", "balls", "runs", "wickets",
              "economy", "dot_pct", "boundary_pct", "strike_rate"]].copy()
    tdf = tdf.rename(columns={
        "seq": "#", "match_date": "Date", "balls": "Balls", "runs": "Runs",
        "wickets": "Wkts", "economy": "Econ", "dot_pct": "Dot %",
        "boundary_pct": "Bdry %", "strike_rate": "SR",
    })
    st.dataframe(
        tdf.style.format({
            "Econ": "{:.2f}", "Dot %": "{:.1f}", "Bdry %": "{:.1f}",
            "SR": "{:.1f}", "Balls": "{:.0f}", "Runs": "{:.0f}", "Wkts": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------------------------------- #
# Coaching report — synthesis & recommendations
#
# This pulls every analysis together for the one selected bowler and turns it
# into a *prioritised plan of action*: each weakness becomes a recommendation
# with a finding, a prescribed drill/decision, and a severity that ranks it
# against the others. The coach leaves with an ordered to-do list, not a wall
# of numbers to interpret.
# --------------------------------------------------------------------------- #
def _fmt_fp(metric: str, v) -> str:
    if pd.isna(v):
        return "—"
    if metric == "economy":
        return f"{v:.2f} rpo"
    if metric == "strike_rate":
        return f"{v:.1f} balls/wkt"
    return f"{v:.1f}%"


# What to say when each fingerprint metric is weak.
_FP_ADVICE = {
    "economy": ("Economy is leaking runs",
        "Build a repeatable over template (stock ball plus one variation) and "
        "drill dot-ball pressure early in the over so he isn't forced to chase."),
    "strike_rate": ("Not taking enough wickets",
        "Add a genuine wicket ball (yorker or hard length into the surface) and "
        "a planned set-up sequence so he attacks for the breakthrough, not just "
        "contains."),
    "dot_pct": ("Too few dot balls",
        "Drill a repeatable defensive length and line to starve the batter — dot "
        "pressure is the cheapest route to wickets."),
    "boundary_pct": ("Conceding too many boundaries",
        "Length-discipline work: cut out the half-volley and the short-and-wide "
        "ball that are being put away."),
}


def build_recommendations(sel_years, sel_countries, bowler, *, fp_min, role_min,
                          exec_minovers, traj_window, traj_min_matches):
    """Synthesise every analysis for one bowler into a ranked recommendation
    list. Returns (recs, positives, facts).

    Each rec is a dict: priority (float), area, title, finding, action. Higher
    priority = more urgent; a live regression or a deployment mismatch outranks
    a mild percentile gap."""
    yrs, ctrs = tuple(sel_years), tuple(sel_countries)
    recs: list[dict] = []
    positives: list[str] = []
    facts: dict = {}

    # ---- Performance fingerprint (all overs) ---------------------------------
    fp = build_table(None, fp_min, yrs, ctrs)
    if bowler in fp.index:
        r = fp.loc[bowler]
        facts["fingerprint"] = r
        for m, (title, action) in _FP_ADVICE.items():
            pct = r.get(f"{m}_pctile")
            if pd.notna(pct) and pct < 45:
                recs.append(dict(
                    priority=float(50 - pct), area="Performance", title=title,
                    finding=f"{FINGERPRINT_METRICS[m]['label']} sits at the "
                            f"{pct:.0f}th percentile vs peers "
                            f"({_fmt_fp(m, r[m])}).",
                    action=action))
            elif pd.notna(pct) and pct >= 75:
                positives.append(
                    f"{FINGERPRINT_METRICS[m]['label']} is a strength "
                    f"({pct:.0f}th pct)")

    # ---- Role / deployment ---------------------------------------------------
    try:
        roles = build_roles(role_min, 30, yrs, ctrs)
    except Exception:
        roles = None
    if roles is not None and bowler in roles.index:
        rr = roles.loc[bowler]
        facts["role"] = rr
        if not bool(rr["in_role"]):
            arch_p = rr["archetype_phase"]
            dep_p = rr["deployment_phase"]
            dep_share = rr[f"{dep_p}_share"]
            gap = float(rr[arch_p] - rr[dep_p])
            recs.append(dict(
                priority=float(max(gap, 15.0)), area="Deployment",
                title=f"Deployment mismatch — best in the {arch_p}",
                finding=f"Profiles as a {rr['archetype']} (strongest in the "
                        f"{arch_p}, skill {rr[arch_p]:.0f}/100) but bowls most of "
                        f"his overs in the {dep_p} ({dep_share:.0f}% of them).",
                action=f"Trial him more in the {arch_p}, where his skill profile "
                       f"is strongest, and track economy and wickets there over "
                       f"the next block of matches."))
        else:
            positives.append(
                f"Correctly deployed as a {rr['archetype']} "
                f"in the {rr['deployment_phase']}")

    # ---- Execution consistency ----------------------------------------------
    cons, _ot = build_consistency(exec_minovers, yrs, ctrs)
    if bowler in cons.index:
        cr = cons.loc[bowler]
        facts["consistency"] = cr
        pctile = cr.get("consistency_pctile")
        if pd.notna(pctile) and pctile < 45:
            in_coh = cons[cons["in_cohort"]]
            best_std = in_coh["std_runs"].min() if not in_coh.empty else np.nan
            gap = cr["std_runs"] - best_std if pd.notna(best_std) else np.nan
            action = ("Repeatability blocks: bowl 6-ball sets to a single plan and "
                      "log the runs per set; ")
            action += (f"goal is to close the {gap:.2f}-run gap to the cohort's "
                       f"tightest bowler." if pd.notna(gap)
                       else "goal is to tighten the over-to-over spread.")
            recs.append(dict(
                priority=float(50 - pctile), area="Execution",
                title="Over-to-over execution is inconsistent",
                finding=f"Consistency at the {pctile:.0f}th percentile "
                        f"(runs-per-over spread {cr['std_runs']:.2f}).",
                action=action))
        elif pd.notna(pctile) and pctile >= 75:
            positives.append(f"Highly repeatable overs ({pctile:.0f}th pct)")

    # ---- Runs leak & matchups ------------------------------------------------
    prof = build_profile_frame(yrs, ctrs)
    leak = runs_leak_breakdown(prof, bowler)
    if not leak.empty:
        facts["leak"] = leak
        bdry = float(leak.loc[leak["source"].str.startswith("Boundaries"), "pct"].iloc[0])
        extras = float(leak.loc[leak["source"].str.startswith("Extras"), "pct"].iloc[0])
        if bdry >= 45:
            recs.append(dict(
                priority=float(bdry - 40), area="Execution",
                title="Runs leak mainly through the boundary",
                finding=f"{bdry:.0f}% of the runs he concedes come in boundaries.",
                action="Length and variation under pressure: deny width and the "
                       "hittable length, and add a change-up to break the batter's "
                       "timing."))
        if extras >= 8:
            recs.append(dict(
                priority=float(extras * 1.5), area="Discipline",
                title="Extras are costing more than they should",
                finding=f"{extras:.0f}% of his conceded runs are wides/no-balls.",
                action="Run-up and front-line discipline drills — a no-ball or wide "
                       "is a free run and an extra delivery."))
    mu = opposition_matchup(prof, bowler, min_balls=24)
    if not mu.empty:
        med = float(mu["economy"].median())
        worst = mu.tail(1)
        wname, wecon = worst.index[0], float(worst["economy"].iloc[0])
        if wecon > med + 1.5:
            recs.append(dict(
                priority=float(min((wecon - med) * 4.0, 40.0)), area="Matchup",
                title=f"Gets taken down by {wname}",
                finding=f"Economy {wecon:.2f} rpo against {wname}, well above his "
                        f"{med:.2f} rpo median across opponents.",
                action=f"Build a specific plan for {wname}: review their scoring "
                       f"zones and pre-set fields and lengths before that fixture."))

    # ---- Development trajectory ---------------------------------------------
    frame = build_trajectory_frame(yrs, ctrs)
    tl = bowler_match_timeline(frame, bowler)
    facts["timeline"] = tl
    if len(tl) >= max(traj_window + 2, traj_min_matches):
        summ = trajectory_summary(tl, traj_window)
        facts["trajectory"] = summ
        for m, srow in summ.iterrows():
            meta = TRAJECTORY_METRICS[m]
            if srow["direction"] == "Worsening":
                recs.append(dict(
                    priority=65.0, area="Trajectory",
                    title=f"{meta['label']} is trending the wrong way",
                    finding=f"Rolling {meta['label'].lower()} is worsening "
                            f"({srow['slope_per10']:+.2f} {meta['unit']} per 10 "
                            f"matches).",
                    action="Treat as a live regression: review recent footage and "
                           "workload and intervene before it sets in."))
            elif srow["direction"] == "Improving":
                positives.append(f"{meta['label']} is improving over time")
            # A statistically clear recent step in the wrong direction.
            sp = srow["step_p"]
            if pd.notna(sp) and sp < 0.05:
                before, after = srow["step_before"], srow["step_after"]
                worse = (after > before) if meta["lower_is_better"] else (after < before)
                if worse and pd.notna(before) and pd.notna(after):
                    recs.append(dict(
                        priority=60.0, area="Trajectory",
                        title=f"Recent step-change worsened {meta['label'].lower()}",
                        finding=f"Level shifted from {meta['fmt'].format(before)} to "
                                f"{meta['fmt'].format(after)} {meta['unit']}.",
                        action="Identify what changed around that point (action, "
                               "role or fitness) and correct it."))

    # ---- Rank & label --------------------------------------------------------
    recs.sort(key=lambda d: d["priority"], reverse=True)
    for i, d in enumerate(recs, 1):
        d["rank"] = i
        d["severity"] = ("High" if d["priority"] >= 55
                         else "Medium" if d["priority"] >= 25 else "Low")
    return recs, positives, facts


def _summary_lines(bowler, facts, recs, positives) -> list[str]:
    lines = []
    role = facts.get("role")
    if role is not None:
        verdict = "correctly deployed" if bool(role["in_role"]) else "mis-deployed"
        lines.append(f"{bowler} profiles as a {role['archetype']} and is "
                     f"{verdict}, bowling most of his overs in the "
                     f"{role['deployment_phase']}.")
    if recs:
        top = recs[0]
        n_high = sum(1 for r in recs if r["severity"] == "High")
        lines.append(f"Top priority: {top['title'][0].lower() + top['title'][1:]} "
                     f"— {top['finding']}")
        lines.append(f"{len(recs)} recommendation(s) in total, {n_high} high "
                     f"priority — see the prioritised plan below.")
    else:
        lines.append("No material weaknesses were flagged under these filters — "
                     "maintain the current programme.")
    if positives:
        lines.append("Strengths to protect: " + "; ".join(positives[:3]) + ".")
    return lines


def assemble_report(sel_years, sel_countries, bowler, *, fp_min, role_min,
                    exec_minovers, traj_window, traj_min_matches) -> dict:
    """Everything the on-screen report and the PDF both need, computed once."""
    yrs, ctrs = tuple(sel_years), tuple(sel_countries)
    recs, positives, facts = build_recommendations(
        sel_years, sel_countries, bowler, fp_min=fp_min, role_min=role_min,
        exec_minovers=exec_minovers, traj_window=traj_window,
        traj_min_matches=traj_min_matches)

    # Fingerprint bar data (percentile gap vs the peer median).
    fp_data = None
    fp = build_table(None, fp_min, yrs, ctrs)
    if bowler in fp.index:
        r = fp.loc[bowler]
        keys = list(FINGERPRINT_METRICS)
        pctiles = [r.get(f"{m}_pctile") for m in keys]
        fp_data = dict(
            labels=[FINGERPRINT_METRICS[m]["label"] for m in keys],
            pctiles=pctiles,
            gaps=[(p - 50) if pd.notna(p) else 0.0 for p in pctiles],
            values=[_fmt_fp(m, r[m]) for m in keys],
        )

    # Trajectory improvement-index data (rebased to 100, up = better).
    traj_data = None
    tl = facts.get("timeline")
    if tl is not None and len(tl) >= max(traj_window + 2, traj_min_matches):
        series = {}
        for m, meta in TRAJECTORY_METRICS.items():
            rr = rolling_metric(tl, m, traj_window)
            rv = rr.dropna()
            if rv.empty:
                continue
            base = rv.iloc[0]
            if not np.isfinite(base) or base == 0:
                continue
            idx = (base / rr) * 100 if meta["lower_is_better"] else (rr / base) * 100
            series[meta["label"]] = idx.to_numpy()
        if series:
            traj_data = dict(dates=tl["match_date"], series=series)

    yr_txt = ", ".join(map(str, sel_years)) if sel_years else "all years"
    ctry_txt = ", ".join(sel_countries) if sel_countries else "all teams"
    context_txt = f"{yr_txt} · {ctry_txt} · generated {datetime.now():%d %b %Y}"
    return dict(
        bowler=bowler, context_txt=context_txt,
        summary_lines=_summary_lines(bowler, facts, recs, positives),
        recs=recs, positives=positives, fp=fp_data, traj=traj_data, facts=facts)


def build_report_pdf(report: dict) -> bytes:
    """Render the concise coaching report to a multi-page PDF (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    serif = {"family": "serif"}
    sev_color = {"High": "#C0392B", "Medium": "#E67E22", "Low": "#27AE60"}
    L, Rx = 0.07, 0.93
    buf = io.BytesIO()

    def _safe(t):
        # The default serif PDF font drops a few glyphs; map them to ASCII so
        # nothing silently disappears from the page.
        return (str(t).replace("—", "-").replace("–", "-")
                .replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))

    with PdfPages(buf) as pdf:
        state = {"fig": None, "ax": None, "y": 0.0}

        def new_page():
            if state["fig"] is not None:
                pdf.savefig(state["fig"])
                plt.close(state["fig"])
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            # Lock the coordinate system to 0-1: without this, the divider
            # ax.plot() below autoscales the axes and stretches every text line.
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_autoscale_on(False)
            state["fig"], state["ax"], state["y"] = fig, ax, 0.96

        def emit(text, size=10, indent=0.0, color="#000000", weight="normal",
                 gap=0.004, width=108):
            for seg in (textwrap.wrap(_safe(text), width) or [""]):
                if state["y"] < 0.07:
                    new_page()
                state["ax"].text(L + indent, state["y"], seg, fontsize=size,
                                 color=color, weight=weight, **serif)
                state["y"] -= 0.0172
            state["y"] -= gap

        # ---- Page 1: header, summary, prioritised plan ----
        new_page()
        ax = state["ax"]
        ax.text(L, state["y"], "Bowler Coaching Report", fontsize=20,
                weight="bold", **serif); state["y"] -= 0.026
        ax.text(L, state["y"], _safe(f"{report['bowler']}  ·  {report['context_txt']}"),
                fontsize=9.5, color="#555555", **serif); state["y"] -= 0.028
        ax.plot([L, Rx], [state["y"], state["y"]], color="#999999", lw=0.8)
        state["y"] -= 0.022

        emit("Executive summary", size=13, weight="bold", gap=0.008)
        for line in report["summary_lines"]:
            emit(line, size=10, gap=0.006)
        state["y"] -= 0.010

        emit("Prioritised plan of action", size=13, weight="bold", gap=0.010)
        if not report["recs"]:
            emit("No material weaknesses flagged — maintain the current "
                 "programme.", size=10)
        for rec in report["recs"]:
            if state["y"] < 0.14:
                new_page()
            sev = rec["severity"]
            state["ax"].text(L, state["y"],
                             _safe(f"{rec['rank']}. {rec['title']}"), fontsize=11.5,
                             weight="bold", **serif)
            state["ax"].text(Rx, state["y"], f"{sev} priority", fontsize=9,
                             weight="bold", color=sev_color[sev], ha="right",
                             **serif)
            state["y"] -= 0.022
            emit(f"Finding — {rec['finding']}", size=9.5, indent=0.012,
                 color="#333333", gap=0.002)
            emit(f"Action — {rec['action']}", size=9.5, indent=0.012,
                 gap=0.012)

        if report["positives"]:
            if state["y"] < 0.16:
                new_page()
            emit("Strengths to protect", size=12, weight="bold", gap=0.008)
            for p in report["positives"]:
                emit(f"•  {p}", size=9.5, indent=0.012, gap=0.002)

        # ---- Charts page ----
        if report["fp"] or report["traj"]:
            if state["fig"] is not None:
                pdf.savefig(state["fig"]); plt.close(state["fig"]); state["fig"] = None
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
            fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.28,
                                left=0.16, right=0.94)
            fig.suptitle(f"{report['bowler']} — key visuals", fontsize=14,
                         weight="bold", family="serif")

            ax1 = axes[0]
            if report["fp"]:
                fpd = report["fp"]
                gaps = fpd["gaps"]
                colors = ["#27AE60" if g >= 0 else "#E74C3C" for g in gaps]
                ax1.barh(fpd["labels"], gaps, color=colors)
                ax1.axvline(0, color="#999999", lw=0.8)
                ax1.set_xlim(-55, 55)
                ax1.invert_yaxis()
                ax1.set_title("Percentile gap vs peer median  (right = better)",
                              fontsize=10, family="serif")
                for yi, (g, v) in enumerate(zip(gaps, fpd["values"])):
                    ax1.text(g + (2 if g >= 0 else -2), yi, v, va="center",
                             ha="left" if g >= 0 else "right", fontsize=8,
                             family="serif")
                ax1.tick_params(labelsize=9)
            else:
                ax1.axis("off")
                ax1.text(0.5, 0.5, "Fingerprint not available under these filters",
                         ha="center", family="serif")

            ax2 = axes[1]
            if report["traj"]:
                tr = report["traj"]
                for lab, vals in tr["series"].items():
                    ax2.plot(tr["dates"], vals, lw=1.6, label=lab)
                ax2.axhline(100, color="#999999", ls="--", lw=0.8)
                ax2.set_title("Improvement index over time  (start = 100, up = "
                              "better)", fontsize=10, family="serif")
                ax2.legend(fontsize=7, ncol=2)
                ax2.tick_params(labelsize=7)
            else:
                ax2.axis("off")
                ax2.text(0.5, 0.5, "Trajectory not available under these filters",
                         ha="center", family="serif")
            pdf.savefig(fig); plt.close(fig)
        elif state["fig"] is not None:
            pdf.savefig(state["fig"]); plt.close(state["fig"])

        info = pdf.infodict()
        info["Title"] = f"Coaching Report — {report['bowler']}"
        info["Subject"] = "T20 bowler coaching report"

    return buf.getvalue()


def render_report(sel_years, sel_countries, bowler, *, fp_min, role_min,
                  exec_minovers, traj_window, traj_min_matches):
  st.subheader("Coaching Report")
  st.caption(
    "A single synthesised read on the selected bowler — where he stands, where "
    "he's heading, and a prioritised plan of action you can hand him. Download "
    "it as a PDF at the bottom."
  )

  report = assemble_report(
      sel_years, sel_countries, bowler, fp_min=fp_min, role_min=role_min,
      exec_minovers=exec_minovers, traj_window=traj_window,
      traj_min_matches=traj_min_matches)

  st.markdown(f"### {bowler}")
  st.caption(report["context_txt"])

  # --- Executive summary ----------------------------------------------------
  st.info("\n\n".join(report["summary_lines"]))

  green, red, grey, blue = "#27AE60", "#E74C3C", "#9aa0a6", "#2E86DE"

  # --- Key visuals ----------------------------------------------------------
  c1, c2 = st.columns([1, 1])
  with c1:
    st.markdown("**Where he stands vs peers**")
    if report["fp"]:
      fpd = report["fp"]
      colors = [green if g >= 0 else red for g in fpd["gaps"]]
      bar = go.Figure(go.Bar(
          x=fpd["gaps"], y=fpd["labels"], orientation="h", marker_color=colors,
          text=[f"{v}  ({p:.0f}th)" if pd.notna(p) else "—"
                for v, p in zip(fpd["values"], fpd["pctiles"])],
          textposition="outside",
          hovertemplate="<b>%{y}</b><br>%{x:+.0f} vs median<extra></extra>",
      ))
      bar.add_vline(x=0, line_color=grey)
      bar.update_layout(
          font=dict(family="Georgia, serif"),
          xaxis=dict(range=[-60, 60], title="Percentile gap from peer median"),
          yaxis=dict(autorange="reversed"),
          height=320, margin=dict(l=10, r=10, t=10, b=30), showlegend=False)
      st.plotly_chart(bar, use_container_width=True)
    else:
      st.caption("Fingerprint not available under these filters.")
  with c2:
    st.markdown("**Where he's heading**")
    if report["traj"]:
      tr = report["traj"]
      palette = ["#2E86DE", "#27AE60", "#E67E22", "#8E44AD"]
      idx_fig = go.Figure()
      for (lab, vals), col in zip(tr["series"].items(), palette):
        idx_fig.add_trace(go.Scatter(
            x=tr["dates"], y=vals, mode="lines", name=lab,
            line=dict(width=2, color=col),
            hovertemplate=lab + ": %{y:.0f}<extra></extra>"))
      idx_fig.add_hline(y=100, line_color=grey, line_dash="dash")
      idx_fig.update_layout(
          font=dict(family="Georgia, serif"),
          yaxis=dict(title="Improvement index (start = 100)"),
          xaxis=dict(title="Match date"),
          height=320, legend=dict(orientation="h", y=-0.3),
          margin=dict(l=10, r=10, t=10, b=30))
      st.plotly_chart(idx_fig, use_container_width=True)
    else:
      st.caption("Trajectory not available under these filters.")

  st.caption("Left: how far above (green) or below (red) the peer median he "
             "sits on each metric. Right: each metric rebased to 100 and flipped "
             "so a rising line means development is showing up.")

  st.divider()

  # --- Prioritised plan of action -------------------------------------------
  st.subheader("Prioritised plan of action")
  if not report["recs"]:
    st.success("No material weaknesses flagged under these filters — maintain "
               "the current programme and protect his strengths.")
  else:
    sev_emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    for rec in report["recs"]:
      st.markdown(
        f"**{rec['rank']}. {rec['title']}**  "
        f"&nbsp;{sev_emoji[rec['severity']]} _{rec['severity']} priority · "
        f"{rec['area']}_"
      )
      st.markdown(f"- **Finding:** {rec['finding']}")
      st.markdown(f"- **Action:** {rec['action']}")

  if report["positives"]:
    st.markdown("**Strengths to protect**")
    st.markdown("\n".join(f"- {p}" for p in report["positives"]))

  # --- PDF export -----------------------------------------------------------
  st.divider()
  st.subheader("Export")
  if not _HAVE_MPL:
    st.info("PDF export needs matplotlib. Install it with "
            "`pip install matplotlib`, then reload.")
  else:
    try:
      pdf_bytes = build_report_pdf(report)
      st.download_button(
          "⬇  Download PDF report", data=pdf_bytes,
          file_name=f"{bowler.replace(' ', '_')}_coaching_report.pdf",
          mime="application/pdf")
    except Exception as exc:  # pragma: no cover - surfaced to the user
      st.error(f"Could not build the PDF: {exc}")


# --------------------------------------------------------------------------- #
# Shared sidebar filters + bowler selection (common to every tab)
# --------------------------------------------------------------------------- #
st.title("T20 Bowler Coaching Tool")

st.session_state.setdefault("flt_year", [])
st.session_state.setdefault("flt_country", [])
st.session_state.setdefault("flt_fp_min", 120)
st.session_state.setdefault("flt_role_min", 120)
st.session_state.setdefault("flt_exec_minovers", 30)
st.session_state.setdefault("flt_traj_window", 5)
st.session_state.setdefault("flt_traj_minmatches", 12)

with st.sidebar:
    st.header("Filters")
    st.caption("These apply to every tab — pick a bowler once and his whole "
               "report follows.")
    with st.form("shared_filters"):
        st.multiselect("Year", all_years, key="flt_year",
                       help="Leave empty for all years.")
        st.multiselect("Country (bowling team)", all_countries, key="flt_country",
                       help="Leave empty for all teams.")
        with st.expander("Advanced cohort thresholds"):
            st.slider("Fingerprint cohort — min legal balls", 30, 600, step=30,
                      key="flt_fp_min")
            st.slider("Role cohort — min legal balls (overall)", 60, 1200,
                      step=60, key="flt_role_min")
            st.slider("Consistency cohort — min overs", 10, 300, step=10,
                      key="flt_exec_minovers")
            st.slider("Trajectory — rolling window (matches)", 3, 15, step=1,
                      key="flt_traj_window")
            st.slider("Trajectory — min matches", 6, 60, step=2,
                      key="flt_traj_minmatches")
        st.form_submit_button("Apply filters")

# Values reflect the last *applied* filters (a form only updates these on submit).
sel_years = st.session_state["flt_year"]
sel_countries = st.session_state["flt_country"]
fp_min = st.session_state["flt_fp_min"]
role_min = st.session_state["flt_role_min"]
exec_minovers = st.session_state["flt_exec_minovers"]
traj_window = st.session_state["flt_traj_window"]
traj_minmatches = st.session_state["flt_traj_minmatches"]

# Shared bowler universe: bowlers clearing the fingerprint cohort under the
# applied year/country filters. The picker sits outside the form so switching
# bowler is instant.
master = build_table(None, fp_min, tuple(sel_years), tuple(sel_countries))
master_cohort = master[master["in_cohort"]]
with st.sidebar:
    if master_cohort.empty:
        st.warning("No bowlers match these filters / fingerprint min-balls. "
                   "Widen the year or country filter, or lower the threshold.")
        bowler = None
    else:
        bowler = stable_bowler_selectbox(
            "Bowler", sorted(master_cohort.index.tolist()), "shared_bowler")

if not bowler:
    st.info("Choose filters and a bowler in the sidebar to see his coaching "
            "report across every tab.")
    st.stop()

# --------------------------------------------------------------------------- #
# Render — one bowler, every lens, then the synthesised report
# --------------------------------------------------------------------------- #
tab_fp, tab_role, tab_exec, tab_traj, tab_report = st.tabs(
    ["Performance Fingerprint", "Role Identification", "Execution & Matchups",
     "Development Trajectory", "Coaching Report"]
)
with tab_fp:
    render_fingerprint(sel_years, sel_countries, bowler, fp_min)
with tab_role:
    render_roles(sel_years, sel_countries, bowler, role_min)
with tab_exec:
    render_execution(sel_years, sel_countries, bowler, exec_minovers)
with tab_traj:
    render_trajectory(sel_years, sel_countries, bowler, traj_window, traj_minmatches)
with tab_report:
    render_report(sel_years, sel_countries, bowler, fp_min=fp_min, role_min=role_min,
                  exec_minovers=exec_minovers, traj_window=traj_window,
                  traj_min_matches=traj_minmatches)