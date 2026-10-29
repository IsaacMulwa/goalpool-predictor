"""Layer 1: Elo ratings replayed over every finished match."""
import math

START = 1500.0
K = 20.0
HOME_ADVANTAGE = 60.0
DRAW_PEAK = 0.28
DRAW_DECAY = 300.0


def expected_home(elo_home, elo_away):
    """Home side's expected score (win=1, draw=0.5), including home advantage."""
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + HOME_ADVANTAGE)) / 400.0))


def build_ratings(matches):
    """matches: (utc_date, home_id, away_id, home_goals, away_goals) tuples. Returns {team_id: elo}."""
    ratings = {}
    for _, home, away, home_goals, away_goals in sorted(matches, key=lambda m: m[0]):
        r_home = ratings.get(home, START)
        r_away = ratings.get(away, START)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0
        delta = K * (actual - expected_home(r_home, r_away))
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return ratings


def elo_probs(elo_home, elo_away):
    """1X2 probabilities. The draw share peaks at 0.28 when the two ratings are level
    and decays with the raw rating gap; the rest is split by the home win expectation."""
    draw = DRAW_PEAK * math.exp(-abs(elo_home - elo_away) / DRAW_DECAY)
    expected = expected_home(elo_home, elo_away)
    return {
        "home": (1.0 - draw) * expected,
        "draw": draw,
        "away": (1.0 - draw) * (1.0 - expected),
    }
