# PhaseGuard — Teammate Setup (10 minutes, no Docker, no local PostgreSQL)

We use a shared cloud database (Neon) so nobody needs to install PostgreSQL
locally. Just Python packages on your machine.

## 0. Install once (if you don't already have these)

- Anaconda: https://www.anaconda.com/download
- Git: https://git-scm.com/downloads

## 1. Clone and enter the project

```bash
git clone <REPO_URL>
cd phaseguard
```

## 2. Create the Python environment

```bash
conda create -n phaseguard python=3.10
conda activate phaseguard
pip install -r requirements.txt
```

This takes 5-10 minutes (downloads PyTorch + SpeechBrain). Get a coffee.

## 3. Set up your .env file

```bash
copy .env.example .env
```

Open `.env` in a text editor. Ask in the team chat for the **real Neon
connection string** and paste it in, replacing the two `DATABASE_URL` /
`DATABASE_URL_SYNC` lines. Leave everything else as-is.

## 4. Run migrations (SKIP if a teammate already did this)

```bash
alembic upgrade head
```

If it says "already at head" — that's fine, it means the tables already
exist in our shared database. Nothing to do.

## 5. Start the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for: `INFO  Application startup complete.`
(First run downloads the ECAPA-TDNN model, ~100MB — needs internet, only
happens once.)

## 6. Start the frontend (open a NEW terminal)

```bash
conda activate phaseguard
cd streamlit_app
streamlit run app.py
```

Your browser opens http://localhost:8501 automatically.

## 7. Verify it works

```bash
pytest tests/ -v
```

Should say `45 passed`.

## You're done

Anyone's enrollment shows up for everyone, since we all share the same
Neon database. No syncing needed.

## If something breaks

| Error | Fix |
|---|---|
| `pip install` fails on some package | Run `pip install <package> --break-system-packages` for that one package, then retry `pip install -r requirements.txt` |
| `Connection refused` / can't reach database | Check `.env` has the real Neon string, not the placeholder `USERNAME:PASSWORD@ep-xxxxx...` |
| `relation "users" does not exist` | Someone needs to run `alembic upgrade head` once |
| Streamlit says "Could not reach API" | Make sure the `uvicorn` terminal from Step 5 is still running |
| Anything else | Paste the exact error in the team chat — don't debug alone, we're on a deadline |
