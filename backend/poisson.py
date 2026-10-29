"""Layer 2: time-decayed Dixon-Coles bivariate Poisson with an exact scoreline grid."""
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

# A match's weight halves every 90 days, so this season's form quickly outweighs last
# season's, which is still included as a fading head start.
HALF_LIFE_DAYS = 90.0
SHRINK = 3.0
RHO = -0.13
MAX_GOALS = 8

_FACTORIALS = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


def _parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def decay_weight(utc_date, now):
    days_ago = max(0.0, (now - _parse_utc(utc_date)).total_seconds() / 86400.0)
    return 0.5 ** (days_ago / HALF_LIFE_DAYS)


def fit(matches, now=None):
    """matches: (utc_date, home_id, away_id, home_goals, away_goals) tuples.

    Each team gets one attack and one defence strength from all its games (home and
    away), as a ratio to the league average (1.0 = average): attack above 1 scores
    more, defence above 1 concedes more (weaker). Home advantage comes from the
    separate league home/away scoring averages."""
    now = now or datetime.now(timezone.utc)

    total_w = total_home_goals = total_away_goals = 0.0
    scored = defaultdict(float)
    conceded = defaultdict(float)
    played = defaultdict(float)
    count = defaultdict(int)

    for utc_date, home, away, hg, ag in matches:
        w = decay_weight(utc_date, now)
        total_w += w
        total_home_goals += w * hg
        total_away_goals += w * ag
        for team, goals_for, goals_against in ((home, hg, ag), (away, ag, hg)):
            scored[team] += w * goals_for
            conceded[team] += w * goals_against
            played[team] += w
            count[team] += 1

    if total_w == 0:
        raise ValueError("No finished matches to fit.")
    league_home_avg = total_home_goals / total_w
    league_away_avg = total_away_goals / total_w
    if league_home_avg == 0 or league_away_avg == 0:
        raise ValueError("League averages are zero; not enough scoring data to fit.")
    avg = (league_home_avg + league_away_avg) / 2.0

    def strength(goals, games):
        return (goals + SHRINK * avg) / (games + SHRINK) / avg

    return {
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "attack": {t: strength(scored[t], played[t]) for t in played},
        "defence": {t: strength(conceded[t], played[t]) for t in played},
        "count": dict(count),
    }


def team_strengths(model, home, away):
    """The four strengths that feed one fixture. A team with no history gets 1.0,
    which is exactly what the shrinkage formula gives for zero matches played."""
    return {
        "attack_home": model["attack"].get(home, 1.0),
        "defence_away": model["defence"].get(away, 1.0),
        "attack_away": model["attack"].get(away, 1.0),
        "defence_home": model["defence"].get(home, 1.0),
    }


def expected_goals(model, home, away):
    s = team_strengths(model, home, away)
    lambda_home = s["attack_home"] * s["defence_away"] * model["league_home_avg"]
    lambda_away = s["attack_away"] * s["defence_home"] * model["league_away_avg"]
    return lambda_home, lambda_away


def _tau(x, y, lambda_home, lambda_away, rho):
    if x == 0 and y == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_home * rho
    if x == 1 and y == 0:
        return 1.0 + lambda_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid(lambda_home, lambda_away, rho=RHO):
    """grid[x][y] = P(home scores x, away scores y) for 0..8 each, normalised to sum to 1."""
    k = np.arange(MAX_GOALS + 1)
    p_home = np.exp(-lambda_home) * lambda_home ** k / _FACTORIALS
    p_away = np.exp(-lambda_away) * lambda_away ** k / _FACTORIALS
    grid = np.outer(p_home, p_away)
    for x in (0, 1):
        for y in (0, 1):
            grid[x, y] *= _tau(x, y, lambda_home, lambda_away, rho)
    np.clip(grid, 0.0, None, out=grid)
    return grid / grid.sum()


def summarise(grid, top_n=5):
    """1X2 probabilities and the most likely scorelines from a grid[x][y]."""
    size = grid.shape[0]
    order = np.argsort(grid, axis=None)[::-1][:top_n]
    top = []
    for idx in order:
        x, y = divmod(int(idx), size)
        top.append({"score": f"{x}-{y}", "prob": float(grid[x, y])})
    return {
        "probabilities": {
            "home": float(np.tril(grid, -1).sum()),
            "draw": float(np.trace(grid)),
            "away": float(np.triu(grid, 1).sum()),
        },
        "most_likely_score": top[0]["score"],
        "top_scorelines": top,
    }
