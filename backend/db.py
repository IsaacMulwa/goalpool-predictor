"""SQLite storage for the predictor: one file (predictor.db), stdlib sqlite3, no ORM."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor.db")

UPCOMING_STATUSES = ("SCHEDULED", "TIMED")
PENDING_STATUSES = ("SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "POSTPONED", "SUSPENDED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    short_name TEXT,
    tla        TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id           INTEGER PRIMARY KEY,
    competition  TEXT NOT NULL,
    season       TEXT,
    utc_date     TEXT NOT NULL,
    status       TEXT NOT NULL,
    matchday     INTEGER,
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    home_goals   INTEGER,
    away_goals   INTEGER,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_matches_comp_status_date
    ON matches (competition, status, utc_date);
"""


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Keep the rollback journal and temp tables in memory so the only file
    # this app ever writes is predictor.db itself.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA temp_store=MEMORY")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    """Create tables if they are missing. Safe to call on every startup."""
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert_matches(competition, rows):
    """Insert or update normalised match rows (see data_fetch._normalise). Returns row count."""
    now = _now_iso()
    teams = {}
    for r in rows:
        teams[r["home"]["id"]] = r["home"]
        teams[r["away"]["id"]] = r["away"]

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO teams (id, name, short_name, tla)
            VALUES (:id, :name, :short_name, :tla)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                short_name = excluded.short_name,
                tla = excluded.tla
            """,
            list(teams.values()),
        )
        conn.executemany(
            """
            INSERT INTO matches (id, competition, season, utc_date, status, matchday,
                                 home_team_id, away_team_id, home_goals, away_goals, updated_at)
            VALUES (:id, :competition, :season, :utc_date, :status, :matchday,
                    :home_id, :away_id, :home_goals, :away_goals, :updated_at)
            ON CONFLICT(id) DO UPDATE SET
                competition = excluded.competition,
                season = excluded.season,
                utc_date = excluded.utc_date,
                status = excluded.status,
                matchday = excluded.matchday,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                home_goals = excluded.home_goals,
                away_goals = excluded.away_goals,
                updated_at = excluded.updated_at
            """,
            [
                {
                    "id": r["id"],
                    "competition": competition,
                    "season": r["season"],
                    "utc_date": r["utc_date"],
                    "status": r["status"],
                    "matchday": r["matchday"],
                    "home_id": r["home"]["id"],
                    "away_id": r["away"]["id"],
                    "home_goals": r["home_goals"],
                    "away_goals": r["away_goals"],
                    "updated_at": now,
                }
                for r in rows
            ],
        )
    return len(rows)


def last_refreshed():
    """{competition: last upsert time as a UTC ISO string} for every stored competition."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT competition, MAX(updated_at) FROM matches GROUP BY competition"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def season_is_complete(competition, season):
    """True if the season is stored and none of its matches is still to be played."""
    with connect() as conn:
        finished, pending = conn.execute(
            f"""
            SELECT SUM(status = 'FINISHED'),
                   SUM(status IN ({",".join("?" * len(PENDING_STATUSES))}))
            FROM matches WHERE competition = ? AND season = ?
            """,
            (*PENDING_STATUSES, competition, season),
        ).fetchone()
    return bool(finished) and not pending


def recent_form(competition, season, team_id, limit=5):
    """The team's last `limit` finished matches of the season, newest first."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT m.utc_date, m.home_team_id, m.home_goals, m.away_goals,
                   h.name AS home_name, a.name AS away_name
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition = ? AND m.season = ? AND m.status = 'FINISHED'
              AND ? IN (m.home_team_id, m.away_team_id)
              AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
            ORDER BY m.utc_date DESC, m.id DESC
            LIMIT ?
            """,
            (competition, season, team_id, limit),
        ).fetchall()
    form = []
    for r in rows:
        at_home = r["home_team_id"] == team_id
        goals_for, goals_against = (r["home_goals"], r["away_goals"]) if at_home else (r["away_goals"], r["home_goals"])
        form.append({
            "date": r["utc_date"][:10],
            "venue": "H" if at_home else "A",
            "opponent": r["away_name"] if at_home else r["home_name"],
            "score": f"{goals_for}-{goals_against}",
            "result": "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L",
        })
    return form


def count_matches(competition):
    """All stored matches for a competition, any status."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE competition = ?", (competition,)
        ).fetchone()
    return row[0]


def current_season(competition):
    """Start year (e.g. '2026') of the newest season stored for a competition, or None."""
    with connect() as conn:
        row = conn.execute(
            "SELECT MAX(season) FROM matches WHERE competition = ?", (competition,)
        ).fetchone()
    return row[0]


def finished_matches(competition, season, before=None):
    """Finished matches of one season as (utc_date, home_id, away_id, home_goals, away_goals),
    oldest first; only those kicking off before `before` (a UTC ISO string), if given."""
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT utc_date, home_team_id, away_team_id, home_goals, away_goals
            FROM matches
            WHERE competition = ? AND season = ? AND status = 'FINISHED'
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
              {"AND utc_date < ?" if before else ""}
            ORDER BY utc_date, id
            """,
            (competition, season, before) if before else (competition, season),
        ).fetchall()
    return [tuple(r) for r in rows]


def matches_between(competition, start, end):
    """Every stored match (any status) kicking off in [start, end), UTC ISO strings, soonest first."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.season, m.utc_date, m.status, m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals, h.name AS home, a.name AS away
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition = ? AND m.utc_date >= ? AND m.utc_date < ?
            ORDER BY m.utc_date, m.id
            """,
            (competition, start, end),
        ).fetchall()
    return [dict(r) for r in rows]


def teams_in_competition(competition, season):
    """Every team that appears in a stored match of the given season."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT t.id, t.name, t.short_name, t.tla
            FROM teams t
            JOIN matches m ON t.id IN (m.home_team_id, m.away_team_id)
            WHERE m.competition = ? AND m.season = ?
            ORDER BY t.name
            """,
            (competition, season),
        ).fetchall()
    return [dict(r) for r in rows]


def upcoming_fixtures(competition):
    """Scheduled matches that have not kicked off yet, soonest first."""
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT m.id, m.utc_date, m.matchday, m.status,
                   h.name AS home, a.name AS away
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition = ?
              AND m.status IN ({",".join("?" * len(UPCOMING_STATUSES))})
              AND m.utc_date >= ?
            ORDER BY m.utc_date, m.id
            """,
            (competition, *UPCOMING_STATUSES, _now_iso()),
        ).fetchall()
    return [dict(r) for r in rows]
