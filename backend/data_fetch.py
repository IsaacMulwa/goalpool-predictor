"""football-data.org v4 client. One API call per competition per refresh."""
import os
import threading
import time

import requests

import db

BASE_URL = "https://api.football-data.org/v4"

COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
}

# Free tier allows 10 requests/minute. Spacing every call by 7 seconds keeps us
# under that, both between competitions in one refresh and across concurrent
# /refresh requests. If the API's throttling headers say the minute's quota is
# already spent (e.g. the key is shared with another client), wait for its reset.
SECONDS_BETWEEN_CALLS = 7.0
REQUEST_TIMEOUT = 30

_call_lock = threading.Lock()
_next_call_at = None  # time.monotonic() value before which no new call may start


class FetchError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def _api_key():
    key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        raise FetchError(
            "FOOTBALL_DATA_API_KEY is not set. Add it to the environment (or .env) and restart.",
            status_code=500,
        )
    return key


def _quota_reset_seconds(resp):
    """Seconds until the API's per-minute counter resets, if the quota is used up, else 0."""
    try:
        available = int(resp.headers.get("X-Requests-Available-Minute", "1"))
        reset = int(resp.headers.get("X-RequestCounter-Reset", "0"))
    except ValueError:
        return 0
    return reset if available <= 0 or resp.status_code == 429 else 0


def fetch_matches(code):
    """Return the raw match list for the competition's current season."""
    global _next_call_at
    key = _api_key()

    with _call_lock:
        if _next_call_at is not None:
            wait = _next_call_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        delay = SECONDS_BETWEEN_CALLS
        try:
            resp = requests.get(
                f"{BASE_URL}/competitions/{code}/matches",
                headers={"X-Auth-Token": key},
                timeout=REQUEST_TIMEOUT,
            )
            delay = max(delay, _quota_reset_seconds(resp) + 1)
        except requests.RequestException as exc:
            raise FetchError(f"{code}: could not reach football-data.org ({exc}).")
        finally:
            _next_call_at = time.monotonic() + delay

    if resp.status_code != 200:
        try:
            message = resp.json().get("message") or resp.text
        except ValueError:
            message = resp.text
        if resp.status_code == 429:
            message = f"rate limit hit; retry in about {int(delay)} seconds. ({message})"
        raise FetchError(f"{code}: football-data.org returned {resp.status_code}: {message}")

    return resp.json().get("matches", [])


def _normalise(match):
    """Flatten one API match into the shape db.upsert_matches expects, or None if unusable."""
    home = match.get("homeTeam") or {}
    away = match.get("awayTeam") or {}
    if not home.get("id") or not away.get("id") or not home.get("name") or not away.get("name"):
        return None
    full_time = (match.get("score") or {}).get("fullTime") or {}
    season_start = (match.get("season") or {}).get("startDate") or ""
    return {
        "id": match["id"],
        "season": season_start[:4] or None,
        "utc_date": match["utcDate"],
        "status": match.get("status") or "UNKNOWN",
        "matchday": match.get("matchday"),
        "home": {"id": home["id"], "name": home["name"],
                 "short_name": home.get("shortName"), "tla": home.get("tla")},
        "away": {"id": away["id"], "name": away["name"],
                 "short_name": away.get("shortName"), "tla": away.get("tla")},
        "home_goals": full_time.get("home"),
        "away_goals": full_time.get("away"),
    }


def refresh(codes):
    """Fetch and upsert each competition's current season. Returns {code: rows_upserted}."""
    counts = {}
    for code in codes:
        rows = [r for r in (_normalise(m) for m in fetch_matches(code)) if r]
        counts[code] = db.upsert_matches(code, rows)
    return counts
