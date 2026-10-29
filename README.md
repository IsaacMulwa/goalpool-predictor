# GoalPool Match Predictor

A free, standalone football match predictor that GoalPool embeds by link or iframe. It shares
nothing with GoalPool's main application: it has its own database, no login, and no session.

- **Backend** (`backend/`): FastAPI app on Render. It fetches real results from football-data.org,
  stores them in SQLite (`predictor.db`), and predicts matches with Elo and a Dixon-Coles Poisson model.
- **Frontend** (`frontend/predictor.html`): one static HTML file with no build step. You upload it
  to GoalPool's hosting by FTP.

Every prediction shows the numbers behind it: Elo ratings, attack and defence strengths, expected
goals, the probabilities from each model, the full 9×9 scoreline grid and a plain-English explanation.

---

## 1. Get a free football-data.org API key

1. Register at <https://www.football-data.org/client/register>.
2. The key (an "API token") is emailed to you and shown on your account page.
3. On the free tier you can make **10 requests per minute**, and you get the major European leagues,
   including the five this tool supports: `PL`, `PD`, `BL1`, `SA` and `FL1`.

## 2. Local setup

Requires Python 3.11.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env              # Windows: copy ..\.env.example .env
# edit .env and set FOOTBALL_DATA_API_KEY
uvicorn main:app --reload
```

The API now runs at <http://localhost:8000>, and interactive docs are at <http://localhost:8000/docs>.

The app looks for `.env` in `backend/` first, then in the repo root. Real environment variables
always win over `.env`.

To try the frontend locally, set `const API = "http://localhost:8000";` in
`frontend/predictor.html`. Then serve the folder so the page has a real origin:

```bash
cd frontend
python -m http.server 5500     # open http://localhost:5500/predictor.html
```

`http://localhost:5500` is already listed in `ALLOWED_ORIGINS` in `.env.example`.

## 3. Seeding data

The database starts empty. Load each competition you want to predict:

```bash
curl -X POST "http://localhost:8000/refresh?competition=PL"
curl -X POST "http://localhost:8000/refresh?competition=PD"
curl -X POST "http://localhost:8000/refresh?competition=BL1"
curl -X POST "http://localhost:8000/refresh?competition=SA"
curl -X POST "http://localhost:8000/refresh?competition=FL1"
```

Or load all five in one call. It takes about 30 seconds because calls are spaced 7 seconds apart:

```bash
curl -X POST "http://localhost:8000/refresh?competition=ALL"
```

- Each competition costs exactly **one** football-data.org request, which returns every match of
  the current season (finished and scheduled).
- The backend waits at least 7 seconds between football-data.org calls, even across separate
  `/refresh` requests, so you can't exceed the free-tier limit.
- Re-running a refresh is safe. Matches are upserted by ID, so results update and new matches are added.
- **Predictions use the current season only.** Last season's results are never used, so a team
  that struggled last year but is flying this year is rated on this year. If older matches are
  still stored, they're ignored, because the model always uses the newest season in the database.
- `/predict` needs at least **30 finished matches** in the current season. That's about 3
  matchdays in PL, PD, SA and FL1 (10 games each) and 4 in BL1 (9 games). Until then, `/predict`
  returns a 400 saying the season has only just started.

## 4. Testing

```bash
curl "http://localhost:8000/health"
# {"status":"ok"}

curl "http://localhost:8000/fixtures?competition=PL"
# {"fixtures":[{"id":...,"utc_date":"2026-11-01T15:00:00Z","matchday":10,"status":"TIMED","home":"Arsenal FC","away":"Chelsea FC"}, ...]}

curl "http://localhost:8000/predict?home=Arsenal&away=Chelsea&competition=PL"
```

The response from `/predict` has this shape. The values shown are illustrative; real values are
computed from your stored matches:

```json
{
  "home": "Arsenal FC",
  "away": "Chelsea FC",
  "probabilities":         {"home": 0.52, "draw": 0.24, "away": 0.24},
  "probabilities_poisson": {"home": 0.53, "draw": 0.25, "away": 0.22},
  "probabilities_elo":     {"home": 0.50, "draw": 0.22, "away": 0.28},
  "most_likely_score": "1-1",
  "top_scorelines": [{"score": "1-1", "prob": 0.11}, {"score": "2-1", "prob": 0.10}],
  "stats_used": {
    "elo_home": 1620, "elo_away": 1580, "elo_diff": 40,
    "attack_home": 1.24, "defence_away": 1.10,
    "attack_away": 1.05, "defence_home": 0.92,
    "lambda_home": 1.63, "lambda_away": 1.15,
    "league_home_avg": 1.52, "league_away_avg": 1.21
  },
  "grid": [[0.04, 0.05, "..."], "... 9 rows of 9 ..."],
  "explanation": ["Based only on this season's form: the 100 PL matches of the 2026/27 season ...", "Elo ratings: ...", "..."]
}
```

- `grid[h][a]` is the probability that the home side scores `h` and the away side scores `a`
  (0 to 8 each). The 81 cells sum to 1.
- Team names are matched without regard to case or accents, against the full name ("Arsenal FC"),
  the short name ("Man City") and the three-letter code ("MUN").
- An unknown team returns **404** with the list of known teams. An ambiguous name such as
  "Manchester" returns **400** with the candidates.
- Fewer than 30 finished matches in the current season returns **400**. If nothing is stored at
  all, the message tells you to call `/refresh` first.

## 5. Deploying to Render

1. Push this folder (`backend/`, `frontend/`, `.env.example`, `README.md`) to a GitHub repo.
   **Do not commit `.env`.** The included `.gitignore` excludes it.
2. In Render, choose **New → Web Service**, connect the repo, and use these settings:

| Setting            | Value                                                  |
|--------------------|--------------------------------------------------------|
| Language / Runtime | Python 3                                               |
| Root Directory     | `backend`                                              |
| Build Command      | `pip install -r requirements.txt`                      |
| Start Command      | `uvicorn main:app --host 0.0.0.0 --port $PORT`         |
| Instance Type      | Free                                                   |
| Health Check Path  | `/health` (under Advanced)                             |

3. Set these environment variables:

| Key                     | Value                                                  |
|-------------------------|--------------------------------------------------------|
| `FOOTBALL_DATA_API_KEY` | your football-data.org token                           |
| `ALLOWED_ORIGINS`       | `https://goalpool.co.ke,https://www.goalpool.co.ke`    |
| `PYTHON_VERSION`        | `3.11.9`                                               |

If `ALLOWED_ORIGINS` is missing or blank, every origin (`*`) is allowed.

4. Deploy. When it's live, `https://<your-service>.onrender.com/health` returns `{"status":"ok"}`.

## 6. Post-deploy: re-seed after every deploy

Render's free tier has **no persistent disk**. `predictor.db` lives on the instance's temporary
filesystem, so it is wiped:

- on **every deploy**, and
- whenever the instance **restarts**, which includes waking up after the free tier's 15-minute idle sleep.

The app recreates empty tables on startup. Until data is loaded again, `/predict` returns a 400
telling you to call `/refresh`. After every deploy (and after any idle sleep) run:

```bash
curl -X POST "https://<your-service>.onrender.com/refresh?competition=ALL"
```

To keep data loaded without manual work, set up a free external scheduler such as
cron-job.org or UptimeRobot:

- `GET https://<your-service>.onrender.com/health` every **10 minutes**. This stops the service
  from sleeping, so the database survives between deploys. One always-on service fits within
  Render's 750 free instance-hours a month.
- `POST https://<your-service>.onrender.com/refresh?competition=ALL` every **6–12 hours**. This
  picks up new results and re-seeds after any restart.

`/refresh` is public by design (there's no authentication in v1). The built-in 7-second spacing
means repeated calls can't break football-data.org's rate limit.

## 7. Embedding in GoalPool

1. Open `frontend/predictor.html` and set the line at the top of the script:

   ```js
   const API = "https://<your-service>.onrender.com";
   ```

2. Upload `predictor.html` to GoalPool's hosting by FTP, for example to
   `public_html/predictor/predictor.html`. You don't need to change any of GoalPool's code or database.
3. Link to it from GoalPool:

   ```html
   <a href="/predictor/predictor.html">Match Predictor</a>
   ```

   Or embed it:

   ```html
   <iframe src="/predictor/predictor.html" title="Match Predictor"
           style="width:100%;max-width:800px;height:1500px;border:0"></iframe>
   ```

4. Deep links prefill the form and run a prediction straight away, which is handy next to a
   fixture on GoalPool:

   ```
   /predictor/predictor.html?competition=PL&home=Arsenal&away=Chelsea
   ```

On first load after the backend has slept, any request that takes longer than 3 seconds shows a
"Waking up…" message. A cold start on Render's free tier takes about a minute.

---

## How the model works (v1)

**Data window: current season only.** Both layers are fitted only on finished matches of the
newest stored season. Every team therefore starts the season level, at Elo 1500 and with
league-average strengths, and is rated purely on what it has done this season.

**Layer 1: Elo.** Every team starts at 1500. All of this season's finished matches are replayed in date order with
K = 20 and a 60-point home advantage. The home side's expected score is
`1 / (1 + 10^((elo_away − (elo_home + 60)) / 400))`, and each result moves both ratings by
`20 × (actual − expected)`. To turn ratings into 1X2 probabilities, the draw probability is
`0.28 × exp(−|elo_home − elo_away| / 300)`, which peaks at 0.28 when the two ratings are equal.
The rest is split by the home side's expected score.

**Layer 2: Dixon-Coles Poisson.** Each match is weighted by `0.5^(days_ago / 180)`. Weighted league
average home and away goals are computed, and from them each team gets four strengths: home attack,
home defence, away attack and away defence. Each strength is shrunk toward 1.0 with 5 pseudo-matches:
`(weighted_scored + 5 × avg) / (weighted_played + 5) / avg`. Expected goals are then
`λ_home = attack_home[home] × defence_away[away] × league_home_avg`, and the mirror image for the
away side. The 0–0 to 8–8 grid is
`Poisson(x, λ_home) × Poisson(y, λ_away) × τ(x, y)`, with the Dixon-Coles correction
(ρ = −0.13) applied to 0-0, 0-1, 1-0 and 1-1. The grid is then normalised to sum to 1. It is
computed exactly; there is no simulation.

**Blend.** Final 1X2 = 0.7 × Poisson + 0.3 × Elo. Scorelines come from the Poisson grid only.

The fitted model is cached in memory per competition. The cache is cleared on every `/refresh`
and rebuilt at least every 6 hours, because the time weights depend on today's date.
