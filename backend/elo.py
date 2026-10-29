"""Layer 1: Elo ratings, carried over from last season and updated by every finished match."""
import math

START = 1500.0
K = 20.0
HOME_ADVANTAGE = 60.0
DRAW_PEAK = 0.28
DRAW_DECAY = 300.0
# Share of last season's rating gap (vs 1500) a team keeps at the start of a new season.
SEASON_CARRY = 2.0 / 3.0


def expected_home(elo_home, elo_away):
    """Home side's expected score (win=1, draw=0.5), including home advantage."""
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + HOME_ADVANTAGE)) / 400.0))


def build_ratings(matches, start=None):
    """matches: (utc_date, home_id, away_id, home_goals, away_goals) tuples.
    start: optional {team_id: rating} to begin from; anyone missing starts at 1500."""
    ratings = dict(start or {})
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


def season_start_ratings(previous_matches, current_teams):
    """Ratings for the start of this season: last season's final Elo regressed toward 1500.
    Promoted teams (not in last season's league) start where the relegated sides finished.
    Returns ({team_id: rating}, set of promoted team ids)."""
    if not previous_matches:
        return {}, set()
    final = build_ratings(previous_matches)
    relegated = [r for team, r in final.items() if team not in current_teams]
    newcomer = sum(relegated) / len(relegated) if relegated else START
    promoted = {t for t in current_teams if t not in final}
    return (
        {t: START + SEASON_CARRY * (final.get(t, newcomer) - START) for t in current_teams},
        promoted,
    )


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
