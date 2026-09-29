# PocketSmart AI: Your Smart Budget & Recommendation Assistant

A GenAI-powered budget planner built with **FastAPI** + **Google Gemini**,
covering three planners: Home Interior, Party, and Jewelry — with login,
session tracking, and recommendation history, matching the project spec.

## Project structure

```
pocketsmart_ai/
├── app.py               # FastAPI app: routes, auth wiring, history
├── auth_utils.py         # JWT auth, password hashing, JSON user store
├── gemini_utils.py       # Gemini prompts + JSON parsing + shopping links
├── requirements.txt
├── .env.example          # copy to .env and fill in your API key
├── static/
│   ├── styles.css
│   └── uploads/          # jewelry outfit images land here
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── home_planner.html
    ├── party_planner.html
    ├── jewelry_planner.html
    ├── recommendations.html
    └── history.html
```

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a Gemini API key

1. Go to https://aistudio.google.com/
2. Sign in, accept terms, click **Get API key**.
3. Copy the key.

## 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=your_real_key_here
SECRET_KEY=some_long_random_string
GEMINI_MODEL=gemini-1.5-flash
```

> Note: the original spec named "Gemini 1.5 Flash Pro." That exact name
> isn't a public Gemini API model id — `gemini-1.5-flash` (or
> `gemini-1.5-pro` for higher quality / higher cost) is the closest
> available equivalent. Change `GEMINI_MODEL` in `.env` to swap models.

## 4. Run it

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

Visit **http://localhost:8000**, register an account, sign in, and try
any of the three planners.

## How it works

- **Auth**: `/register`, `/token` (login), `/logout` — JWT stored in an
  httpOnly cookie, sessions tracked in `active_sessions` and auto-expired
  after 30 minutes of inactivity (`/session-info`, `/session-data`).
- **Home Planner** (`/generate-home`): sends your budget + room/fixture
  counts to Gemini, gets back a structured budget breakdown, and attaches
  real search links to Amazon, Flipkart, and IKEA for each item.
- **Party Planner** (`/generate-party`): sends event details to Gemini,
  gets a category-by-category budget split (catering, decoration,
  entertainment) plus venue suggestions, linked to Swiggy, Zomato,
  BookMyShow, OYO, etc.
- **Jewelry Planner** (`/generate-jewelry`): optionally accepts an outfit
  photo (multimodal input to Gemini) to color-match jewelry suggestions,
  linked to Amazon, Flipkart, BlueStone, Tanishq, CaratLane.
- **History** (`/history`, `/recommendation-details/{id}`): every
  generated plan is saved in memory per user and can be revisited.

## Notes / things to harden for production

- User data lives in `users_db.json` and recommendation history lives in
  memory — swap both for a real database (SQLite/Postgres) if you need
  persistence across restarts or multiple server workers.
- `CORSMiddleware` currently allows all origins — restrict `allow_origins`
  before deploying publicly.
- The shopping "search" links are constructed URLs (e.g.
  `amazon.in/s?k=<term>`), not live product data from partner APIs — the
  spec's mention of "third-party APIs" would require signing up for each
  platform's affiliate/partner program individually.
