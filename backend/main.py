"""GoalPool Match Predictor API: transparent Elo + Dixon-Coles predictions from stored match data."""
import os
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Read-only: a .env next to main.py wins, then one at the repo root. Neither overrides real env vars.
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, os.pardir, ".env"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import data_fetch  # noqa: E402
import db  # noqa: E402
import elo  # noqa: E402
import poisson  # noqa: E402

MIN_FINISHED_MATCHES = 30
POISSON_WEIGHT = 0.7
ELO_WEIGHT = 0.3
CACHE_TTL_SECONDS = 6 * 3600  # time decay is relative to "now", so refit a few times a day
LOW_SAMPLE_MATCHES = 5
# Render's free tier wipes predictor.db whenever the instance restarts (every deploy and every
# wake-up from idle), so the app reloads missing or stale competitions by itself in the
# background. Set AUTO_REFRESH_HOURS=0 to disable and rely on POST /refresh only.
AUTO_REFRESH_HOURS = float(os.getenv("AUTO_REFRESH_HOURS", "6"))
AUTO_REFRESH_CHECK_SECONDS = 60
AUTO_REFRESH_RETRY_SECONDS = 600  # after a failed attempt, don't hammer the API

_cache = {}
_cache_generation = 0
_cache_lock = threading.Lock()

_auto_lock = threading.Lock()
_auto_running = False
_auto_last_check = 0.0
_auto_last_attempt = 0.0


def _stale_competitions():
    cutoff = time.time() - AUTO_REFRESH_HOURS * 3600
    refreshed = db.last_refreshed()
    return [
        code for code in data_fetch.COMPETITIONS
        if code not in refreshed
        or datetime.fromisoformat(refreshed[code].replace("Z", "+00:00")).timestamp() < cutoff
    ]


def _maybe_auto_refresh():
    """Start a background reload of missing/stale competitions, at most one at a time."""
    global _auto_running, _auto_last_check, _auto_last_attempt
    if AUTO_REFRESH_HOURS <= 0 or not os.getenv("FOOTBALL_DATA_API_KEY", "").strip():
        return
    now = time.time()
    with _auto_lock:
        if _auto_running or now - _auto_last_check < AUTO_REFRESH_CHECK_SECONDS:
            return
        _auto_last_check = now
        if now - _auto_last_attempt < AUTO_REFRESH_RETRY_SECONDS:
            return
        codes = _stale_competitions()
        if not codes:
            return
        _auto_running = True
        _auto_last_attempt = now
    threading.Thread(target=_auto_refresh, args=(codes,), daemon=True).start()


def _auto_refresh(codes):
    global _auto_running
    try:
        data_fetch.refresh(codes)
    except data_fetch.FetchError as exc:
        print(f"Auto-refresh failed: {exc}", flush=True)
    finally:
        _clear_cache()
        with _auto_lock:
            _auto_running = False


def _loading_error():
    return HTTPException(
        503, "Match data is loading because the server has just started. Please try again in about a minute."
    )


@asynccontextmanager
async def lifespan(_app):
    db.init()
    _maybe_auto_refresh()
    yield


app = FastAPI(title="GoalPool Match Predictor", version="1.0.0", lifespan=lifespan)


def _allowed_origins():
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- helpers

def _competition(code):
    code = code.strip().upper()
    if code not in data_fetch.COMPETITIONS:
        supported = ", ".join(data_fetch.COMPETITIONS)
        raise HTTPException(400, f"Unknown competition '{code}'. Supported: {supported}.")
    return code


def _refresh_hint(comp):
    return f"The operator must call POST /refresh?competition={comp} first."


def _season_label(season):
    """'2026' -> '2026/27'."""
    start = int(season)
    return f"{start}/{(start + 1) % 100:02d}"


def _clear_cache():
    global _cache_generation
    with _cache_lock:
        _cache.clear()
        _cache_generation += 1


def _get_model(comp):
    """Fitted Elo + Poisson for a competition, cached until the next refresh or TTL expiry.

    Only the current (newest stored) season is used, so last season's form never
    carries over: every team starts the season at Elo 1500 with average strengths."""
    with _cache_lock:
        entry = _cache.get(comp)
        if entry and time.time() - entry["built_at"] < CACHE_TTL_SECONDS:
            return entry
        generation = _cache_generation

    season = db.current_season(comp)
    if season is None:
        if _auto_running:
            raise _loading_error()
        raise HTTPException(400, f"No {comp} matches are stored. {_refresh_hint(comp)}")

    matches = db.finished_matches(comp, season)
    if len(matches) < MIN_FINISHED_MATCHES:
        raise HTTPException(
            400,
            f"Only {len(matches)} {comp} matches of the {_season_label(season)} season have been "
            f"played so far; at least {MIN_FINISHED_MATCHES} are needed. If the season has just "
            f"started, predictions open after a few more matchdays; otherwise the operator should "
            f"call POST /refresh?competition={comp}.",
        )

    entry = {
        "built_at": time.time(),
        "season": season,
        "match_count": len(matches),
        "first_date": matches[0][0][:10],
        "last_date": matches[-1][0][:10],
        "elo": elo.build_ratings(matches),
        "poisson": poisson.fit(matches),
        "teams": db.teams_in_competition(comp, season),
    }
    with _cache_lock:
        # Don't store a model built from data a concurrent /refresh has since replaced.
        if generation == _cache_generation:
            _cache[comp] = entry
    return entry


def _fold(text):
    """Case- and accent-insensitive form, so 'munchen' finds 'FC Bayern München'."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


# Common nicknames that don't appear in football-data.org's name, short name or code.
ALIASES = {
    "spurs": "tottenham",
    "wolves": "wolverhampton",
    "man utd": "manchester united",
    "boro": "middlesbrough",
}


def _core_name(name):
    """Name without leading/trailing club-type tokens: 'FC Barcelona' -> 'barcelona',
    '1. FC Köln' -> 'koln', 'Bologna FC 1909' -> 'bologna'."""
    words = _fold(name).replace(".", " ").split()
    while len(words) > 1 and (len(words[0]) <= 3 or words[0].isdigit()):
        words.pop(0)
    while len(words) > 1 and (len(words[-1]) <= 3 or words[-1].isdigit()):
        words.pop()
    return " ".join(words)


def _resolve_team(query, teams, comp, side):
    q = _fold(query)
    if not q:
        raise HTTPException(400, f"Please enter a {side} team.")
    q = ALIASES.get(q, q)

    exact = [
        t for t in teams
        if q in (_fold(t["name"]), _fold(t["short_name"]), _fold(t["tla"]), _core_name(t["name"]))
    ]
    if len(exact) == 1:
        return exact[0]

    partial = exact or [t for t in teams if q in _fold(t["name"]) or q in _fold(t["short_name"])]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        known = ", ".join(t["name"] for t in teams)
        raise HTTPException(404, f"No {comp} team matches '{query.strip()}'. Known teams: {known}.")
    names = ", ".join(t["name"] for t in partial)
    raise HTTPException(400, f"'{query.strip()}' matches several {comp} teams: {names}. Please be more specific.")


def _pct(p):
    return f"{p * 100:.0f}%"


def _versus_average(value, verb_more, verb_less):
    diff = round((value - 1.0) * 100)
    if abs(diff) < 3:
        return "at about the league-average rate"
    return f"{abs(diff)}% {verb_more if diff > 0 else verb_less} goals than average"


def _explain(comp, model, home, away, stats, p_poisson, p_elo, p_final, top):
    h, a = home["name"], away["name"]
    pm = model["poisson"]
    lines = [
        f"Based only on this season's form: the {model['match_count']} {comp} matches of the "
        f"{_season_label(model['season'])} season played so far ({model['first_date']} to "
        f"{model['last_date']}). Last season's results are not used. Recent matches weigh more: a "
        f"match played {int(poisson.HALF_LIFE_DAYS)} days ago counts half as much as one played today.",
    ]

    diff = stats["elo_diff"]
    if diff == 0:
        gap = "The two sides are rated level"
    else:
        leader = h if diff > 0 else a
        gap = f"{leader} is {abs(diff)} points stronger"
    lines.append(
        f"Elo ratings: {h} {stats['elo_home']}, {a} {stats['elo_away']}. {gap}. Adding the "
        f"{int(elo.HOME_ADVANTAGE)}-point home advantage for {h}, Elo alone says "
        f"{_pct(p_elo['home'])} home / {_pct(p_elo['draw'])} draw / {_pct(p_elo['away'])} away."
    )

    lines.append(
        f"At home, {h} score {_versus_average(stats['attack_home'], 'more', 'fewer')} "
        f"(attack {stats['attack_home']:.2f}), while away from home {a} concede "
        f"{_versus_average(stats['defence_away'], 'more', 'fewer')} (defence {stats['defence_away']:.2f}; "
        f"above 1.00 means leakier than average)."
    )
    lines.append(
        f"Away from home, {a} score {_versus_average(stats['attack_away'], 'more', 'fewer')} "
        f"(attack {stats['attack_away']:.2f}), while at home {h} concede "
        f"{_versus_average(stats['defence_home'], 'more', 'fewer')} (defence {stats['defence_home']:.2f})."
    )

    thin = [
        f"{team} {count} {where} match{'es' if count != 1 else ''}"
        for team, count, where in (
            (h, pm["home_count"].get(home["id"], 0), "home"),
            (a, pm["away_count"].get(away["id"], 0), "away"),
        )
        if count < LOW_SAMPLE_MATCHES
    ]
    if thin:
        lines.append(
            f"Small sample so far this season ({' and '.join(thin)}), so these strengths are "
            f"pulled toward the league average until more games are played."
        )

    lines.append(
        f"Multiplying these by the league averages ({stats['league_home_avg']:.2f} home goals and "
        f"{stats['league_away_avg']:.2f} away goals per match) gives expected goals of "
        f"{stats['lambda_home']:.2f} for {h} and {stats['lambda_away']:.2f} for {a}."
    )
    lines.append(
        f"The Dixon-Coles scoreline model turns those into {_pct(p_poisson['home'])} home win, "
        f"{_pct(p_poisson['draw'])} draw, {_pct(p_poisson['away'])} away win. Blending "
        f"{int(POISSON_WEIGHT * 100)}% of that with {int(ELO_WEIGHT * 100)}% of the Elo view gives "
        f"the final {_pct(p_final['home'])} / {_pct(p_final['draw'])} / {_pct(p_final['away'])}."
    )

    best = top[0]
    covered = sum(s["prob"] for s in top)
    lines.append(
        f"The single most likely score is {best['score']} at {best['prob'] * 100:.1f}%, but exact "
        f"scores are hard to call: the top {len(top)} scorelines together cover only {covered * 100:.0f}%."
    )

    favourite = max(p_final, key=p_final.get)
    x, y = (int(g) for g in best["score"].split("-"))
    if favourite != "draw" and x == y:
        lines.append(
            "A draw can be the single most likely exact score even when one side is favoured, "
            "because that side's wins are spread across many different scorelines."
        )

    ranked = sorted(p_final.values(), reverse=True)
    if ranked[0] - ranked[1] < 0.05:
        lines.append("Verdict: too close to call; the top two outcomes are within 5 percentage points.")
    elif favourite == "draw":
        lines.append(f"Verdict: a draw is the most likely outcome ({_pct(p_final['draw'])}).")
    else:
        team = h if favourite == "home" else a
        lines.append(f"Verdict: {team} are favoured to win ({_pct(p_final[favourite])}).")
    return lines


def _round_probs(p):
    return {k: round(v, 4) for k, v in p.items()}


# --------------------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/refresh")
def refresh(
    competition: str = Query("PL", description="PL, PD, BL1, SA, FL1, a comma-separated list, or ALL"),
):
    raw = competition.strip().upper()
    if raw == "ALL":
        codes = list(data_fetch.COMPETITIONS)
    else:
        codes = list(dict.fromkeys(_competition(c) for c in raw.split(",") if c.strip()))
    if not codes:
        raise HTTPException(400, "No competition given.")

    try:
        data_fetch.refresh(codes)
    except data_fetch.FetchError as exc:
        raise HTTPException(exc.status_code, str(exc))
    finally:
        # Earlier competitions in a multi-refresh may have been written before a failure.
        _clear_cache()
    return {"refreshed": ",".join(codes)}


@app.get("/fixtures")
def fixtures(competition: str = "PL"):
    comp = _competition(competition)
    _maybe_auto_refresh()
    if db.count_matches(comp) == 0:
        if _auto_running:
            raise _loading_error()
        raise HTTPException(400, f"No {comp} matches are stored. {_refresh_hint(comp)}")
    return {"fixtures": db.upcoming_fixtures(comp)}


@app.get("/predict")
def predict(home: str, away: str, competition: str = "PL"):
    comp = _competition(competition)
    _maybe_auto_refresh()
    model = _get_model(comp)
    home_team = _resolve_team(home, model["teams"], comp, "home")
    away_team = _resolve_team(away, model["teams"], comp, "away")
    if home_team["id"] == away_team["id"]:
        raise HTTPException(400, f"Home and away both resolve to {home_team['name']}. Pick two different teams.")

    elo_home = model["elo"].get(home_team["id"], elo.START)
    elo_away = model["elo"].get(away_team["id"], elo.START)
    p_elo = elo.elo_probs(elo_home, elo_away)

    pm = model["poisson"]
    strengths = poisson.team_strengths(pm, home_team["id"], away_team["id"])
    lambda_home, lambda_away = poisson.expected_goals(pm, home_team["id"], away_team["id"])
    grid = poisson.scoreline_grid(lambda_home, lambda_away)
    summary = poisson.summarise(grid)
    p_poisson = summary["probabilities"]
    p_final = {k: POISSON_WEIGHT * p_poisson[k] + ELO_WEIGHT * p_elo[k] for k in ("home", "draw", "away")}

    stats = {
        "elo_home": round(elo_home),
        "elo_away": round(elo_away),
        "elo_diff": round(elo_home) - round(elo_away),
        "attack_home": round(strengths["attack_home"], 2),
        "defence_away": round(strengths["defence_away"], 2),
        "attack_away": round(strengths["attack_away"], 2),
        "defence_home": round(strengths["defence_home"], 2),
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "league_home_avg": round(pm["league_home_avg"], 2),
        "league_away_avg": round(pm["league_away_avg"], 2),
    }
    top = [{"score": s["score"], "prob": round(s["prob"], 4)} for s in summary["top_scorelines"]]

    return {
        "home": home_team["name"],
        "away": away_team["name"],
        "probabilities": _round_probs(p_final),
        "probabilities_poisson": _round_probs(p_poisson),
        "probabilities_elo": _round_probs(p_elo),
        "most_likely_score": summary["most_likely_score"],
        "top_scorelines": top,
        "stats_used": stats,
        # grid[home_goals][away_goals], 0..8 each
        "grid": [[round(float(p), 5) for p in row] for row in grid],
        "explanation": _explain(comp, model, home_team, away_team, stats, p_poisson, p_elo, p_final, top),
    }
