<<<<<<< HEAD
# PhaseGuard — Layer 2: Speaker Verification

Production backend for PhaseGuard's **Layer 2** identity-verification engine, built for the UCO Hackathon 2026.

Layer 2 verifies that a live caller is the enrolled customer by comparing a
192-dimensional ECAPA-TDNN speaker embedding extracted from the live audio
against the stored voiceprint via cosine similarity.

---

## Architecture

```
POST /api/v1/enroll
  WAV files (30-50 utterances)
    → temp storage → ECAPA-TDNN embeddings → average
    → voiceprint stored in PostgreSQL (pgvector VECTOR(192))

POST /api/v1/verify
  Live audio + user_id (+ optional layer1_score from Layer 1)
    → ECAPA-TDNN embedding
    → fetch enrolled voiceprint from PostgreSQL
    → cosine similarity
    → identity decision  (threshold: 0.65)
    → Risk Engine        (combines Layer 1 + Layer 2 scores)
    → persist verification_log + risk_log
    → return VerificationResult
```

### Risk Engine rules

| Condition | Decision |
|---|---|
| `layer1_score > 0.70` | `FRAUD_ALERT` |
| `speaker_similarity < 0.65` | `FRAUD_ALERT` |
| otherwise | `CLEAN` |

---

## Project Structure

```
phaseguard/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── enroll.py          POST /api/v1/enroll
│   │   │   ├── verify.py          POST /api/v1/verify
│   │   │   ├── users.py           POST/GET /api/v1/users
│   │   │   └── history.py         GET /api/v1/verification-history, /risk-history
│   │   └── router.py
│   ├── core/
│   │   ├── config.py              pydantic-settings (reads .env)
│   │   └── logging.py             loguru sinks (console + rotating file)
│   ├── database/
│   │   ├── base.py                SQLAlchemy DeclarativeBase
│   │   └── session.py             async engine + get_db() dependency
│   ├── models/
│   │   ├── user.py                users table
│   │   ├── voiceprint.py          voiceprints table (VECTOR(192))
│   │   ├── verification_log.py    verification_logs table
│   │   └── risk_log.py            risk_logs table
│   ├── schemas/                   Pydantic request/response schemas
│   ├── repositories/              DB access layer (no raw SQL in services)
│   ├── services/
│   │   ├── enrollment_service.py  Enrollment orchestration
│   │   ├── verification_service.py Verification + logging
│   │   └── risk_engine.py         Risk scoring rules
│   ├── ml/
│   │   └── ecapa_service.py       SpeechBrain ECAPA-TDNN wrapper + singleton
│   ├── utils/
│   │   ├── audio_utils.py         Audio I/O, temp file management
│   │   └── exceptions.py          Domain exception hierarchy
│   └── main.py                    FastAPI app + lifespan + exception handlers
├── streamlit_app/
│   ├── app.py                     Home page + health check
│   ├── api_client.py              HTTP client wrapping backend calls
│   ├── config.py                  Frontend settings (API URL, thresholds)
│   └── pages/
│       ├── 1_Enrollment.py        Enroll a user's voiceprint
│       ├── 2_Verification.py      Verify live audio + display risk result
│       ├── 3_History.py           Verification + risk history tables/charts
│       └── 4_Risk_Dashboard.py    Risk gauge, trend charts, distribution pie
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py  CREATE TABLE users/voiceprints/…
├── tests/
│   ├── conftest.py
│   ├── test_risk_engine.py          9 unit tests — all rules
│   ├── test_ecapa_service.py        9 unit tests — cosine similarity + averaging
│   ├── test_audio_utils.py         12 unit tests — I/O, extension validation
│   ├── test_enrollment_service.py   4 unit tests — service orchestration
│   └── test_verification_service.py 6 unit tests — service orchestration
├── Dockerfile                       FastAPI backend image
├── Dockerfile.streamlit             Streamlit frontend image
├── docker-compose.yml               db + api + streamlit
├── alembic.ini
├── requirements.txt
├── pytest.ini
└── .env.example
=======
# 🛡️ PhaseGuard

**Real-Time Two-Layer Voice Forensics for Banking Security & Deepfake Interception**  
*UCO Bank Hackathon 2026 — Built by Team Ozymandias*

---

## 📌 Project Overview
PhaseGuard is an advanced, real-time voice verification system designed to protect banking transactions against AI voice cloning fraud. If a criminal uses a cloned voice (e.g. generated via ElevenLabs in under 30 seconds using a 3-second sample) to call a bank, PhaseGuard intercepts and blocks the call in real-time before any funds are moved.

### Why AI Voices Fail: The Frame Boundary Phenomenon
* **Real Human Voice**: Produced by lungs pushing air through vocal cords, shaped smoothly by the mouth and lips. This is a continuous physical process. Measuring the **phase** (position of the sound wave over time) reveals smooth, continuous, organic transitions.
* **AI Voice (Deepfake)**: Synthesized frame-by-frame (typically in 20-millisecond windows) by a neural network vocoder. Because each frame is calculated independently, **sudden phase jumps or discontinuities** occur at frame boundaries. The human ear can't hear these tiny "waterfalls," but computers can measure them precisely.

---

## 📂 Standard Folder Structure
For PhaseGuard to function correctly, maintain the following directory layout:

```text
E:/UCO-HACKATHON/
│
├── data/
│   ├── real_voices/             # Enrolled genuine customer recordings (16kHz WAV mono)
│   │   ├── vishal/              # vishal_001.wav, vishal_002.wav...
│   │   ├── abhinav/             # abhinav_001.wav, abhinav_002.wav...
│   │   ├── aditya/              # aditya_001.wav, aditya_002.wav...
│   │   └── dhruv/               # dhruv_001.wav, dhruv_002.wav...
│   │
│   ├── clones/                  # Deepfaked voice clone clips (16kHz WAV mono)
│   │   ├── vishal/              # vishal_clone_001.wav...
│   │   ├── abhinav/             # abhinav_clone_001.wav...
│   │   ├── aditya/              # aditya_clone_001.wav...
│   │   └── dhruv/               # dhruv_clone_001.wav...
│   │
│   └── processed/               # Standardized training datasets (generated by pipeline)
│       ├── spectrograms.npy     # Compiled 128x128 mel-spectrogram arrays
│       ├── features.npy         # Compiled 5-signal physical features
│       └── labels.npy           # Labels (0 = Real, 1 = Fake)
│
├── models/                      # Saved weights and template databases
│   ├── layer1_mobilenet.pth     # Trained MobileNetV3 CNN weights
│   └── voiceprints.pth          # Enrolled ECAPA-TDNN speaker templates
│
├── pretrained_models/           # Cached SpeechBrain models (downloaded automatically)
│   └── ecapa/
│
├── requirements.txt             # Python dependency packages list
├── generate_simulated_data.py   # Synthesizes mock audio clips for Vishal, Abhinav, etc.
├── preprocess.py                # Audio normalizer and 5-signal physics feature extractor
├── build_dataset.py             # Preprocesses raw wav files and saves .npy matrices
├── train_layer1.py              # PyTorch script training the MobileNetV3 CNN model
├── enroll_voices.py             # SpeechBrain script building speaker embeddings
├── dashboard.py                 # Premium Streamlit web front-end and charts
├── run_app.py                   # Environment-aware app launcher
└── README.md                    # System architecture and user documentation
>>>>>>> f910cc20bed285d385b74db0145c6f1ca13d6633
```

---

<<<<<<< HEAD
## Quick Start (Hackathon Setup — Shared Cloud Database, No Docker)

For the hackathon, we run the backend and frontend locally on each
teammate's laptop, but all connect to **one shared Neon PostgreSQL
database** (with pgvector enabled) instead of each laptop running its own
local PostgreSQL. This avoids per-laptop database installation issues
(pgvector compilation, `pg_config` PATH errors) and Docker disk-space
issues entirely, while still giving everyone the same enrolled voiceprints
and verification history.

See **[SETUP.md](./SETUP.md)** for the fast, copy-paste version of these
steps.

### 1. One-time: set up the shared database (only one teammate does this)

1. Create a free project at **neon.tech**
2. In Neon's SQL editor, run: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Copy the connection string Neon gives you and share it with the team
   privately (do not commit it to GitHub)

### 2. Every teammate: clone and install

```bash
git clone <repo-url>
cd phaseguard
conda create -n phaseguard python=3.10
conda activate phaseguard
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`, replacing the placeholder `DATABASE_URL` / `DATABASE_URL_SYNC`
values with the real Neon connection string shared by the team.

### 3. Run migrations (only needed once, by whoever sets up first)

```bash
alembic upgrade head
```

### 4. Start the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the frontend (new terminal)

```bash
cd streamlit_app
streamlit run app.py
```

Open **http://localhost:8501**.

---

## Docker (optional, for later/production — not used during the hackathon)

The project still includes a full Docker setup (`Dockerfile`,
`Dockerfile.streamlit`, `docker-compose.yml`) for when you want a fully
containerized deployment later. It is **not required** for local
development or the hackathon demo, and avoids the disk-space and
`pg_config` build issues some Windows laptops hit with a from-scratch local
PostgreSQL install.

```bash
docker-compose up --build
```



## Local Development (without Docker)

### Prerequisites

- Python 3.10+
- PostgreSQL 16 with pgvector extension installed
- `libsndfile` system library (for soundfile / audio loading)

```bash
# macOS
brew install libsndfile ffmpeg

# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit DATABASE_URL / DATABASE_URL_SYNC to point to your local Postgres
```

### Run migrations

```bash
alembic upgrade head
```

### Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Streamlit

```bash
cd streamlit_app
streamlit run app.py
```

---

## API Reference

### `POST /api/v1/users`
Create a new enrollable user.
```json
{ "name": "Vishal Kumar", "email": "vishal@example.com" }
```

### `POST /api/v1/enroll`
Enroll a user's voiceprint from multiple recordings.
- Form fields: `user_id` (UUID), `files` (multipart, 1+ WAV/FLAC/MP3)
- Minimum recordings: configurable via `MIN_ENROLLMENT_RECORDINGS` (default 3; use 30-50 for production)

### `POST /api/v1/verify`
Verify a live audio sample against an enrolled voiceprint.
- Form fields: `user_id` (UUID), `file` (single audio), `layer1_score` (float, default 0.0)
- Returns: similarity score, identity decision, risk level

### `GET /api/v1/users/{user_id}`
Fetch a user and their enrollment status.

### `GET /api/v1/verification-history/{user_id}`
Paginated verification attempt history.

### `GET /api/v1/risk-history/{user_id}`
Paginated risk evaluation history.

---

## Running Tests

```bash
pytest tests/ -v
```

**45 tests, 0 failures.** Tests cover:
- Risk engine decision rules (all branches, boundary conditions)
- ECAPA cosine similarity math (6 properties)
- Enrollment averaging and skip-on-failure logic
- Audio utility I/O and extension validation
- Enrollment and verification service orchestration (mocked DB + ML)

---

## Database Schema

```sql
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE voiceprints (
    id              UUID PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    embedding       VECTOR(192) NOT NULL,   -- pgvector column
    recording_count INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE verification_logs (
    id               UUID PRIMARY KEY,
    user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    decision         VARCHAR(32) NOT NULL,   -- 'verified' | 'mismatch'
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE risk_logs (
    id          UUID PRIMARY KEY,
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    risk_score  FLOAT NOT NULL,
    risk_level  VARCHAR(32) NOT NULL,   -- 'CLEAN' | 'FRAUD_ALERT'
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

---

## Key Design Decisions

**Why pgvector instead of .pth files?**
Voiceprints stored in PostgreSQL are persistent across restarts, queryable by SQL, easily backed up, and support pgvector's native cosine similarity operators for future similarity search.

**Why a singleton ECAPA service?**
The SpeechBrain model is ~100MB and takes several seconds to load. The module-level `get_ecapa_service()` singleton ensures it is loaded only once per process (at startup via `lifespan`), not on every request.

**Why repository/service separation?**
Endpoints call services; services call repositories; repositories call SQLAlchemy. Each layer can be tested in isolation — service tests mock repositories, unit tests mock everything below them.

---

## Next Steps (Layers 1 & 3)

- **Layer 1 (AI Voice Detection):** Train MobileNetV3 on ASVspoof 2024 + WaveFake mel-spectrograms; expose as `POST /api/v1/detect-ai`. Pass the returned `layer1_score` to `POST /api/v1/verify`.
- **Layer 3 (Deepfake Detection):** Add a third model (e.g. RawNet3) for waveform-level artifact detection.
- **Risk Engine expansion:** Update `risk_engine.py` to accept `layer3_score` and extend the decision rules.
=======
## 🛠️ Step-by-Step Environment Setup
Teammates can reproduce the exact working environment on Windows using Anaconda and Command Prompt.

### Step 1: Initialize the Conda Environment
Open **Anaconda Prompt** and execute:
```bash
# Create a dedicated virtual environment with Python 3.10
conda create -n phaseguard python=3.10 -y

# Activate the environment
conda activate phaseguard
```

### Step 2: Install Libraries
With the `phaseguard` environment active, install the dependencies listed in `requirements.txt`:
```bash
# Install PyTorch and Audio tools
pip install torch torchaudio torchvision

# Install audio processing utilities
pip install librosa soundfile numpy==1.23.5 scipy

# Install SpeechBrain (Speaker Verification)
pip install speechbrain

# Install HuggingFace Transformers (required by SpeechBrain)
pip install transformers

# Install Streamlit and Plotly (Dashboard)
pip install streamlit plotly

# Install recording libraries
pip install sounddevice pyaudio
```
*(Alternatively, run: `pip install -r requirements.txt`)*

---

## 📓 Working with Anaconda Jupyter Notebooks
If you are developing or testing **Layer 1 (Authenticity Check / Model Training)** inside Jupyter, follow this setup:

### 1. Register the Conda Environment in Jupyter
To make your custom environment visible inside Jupyter Notebook as a selectable kernel, run:
```bash
# Activate your environment
conda activate phaseguard

# Install ipykernel (Jupyter connector)
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=phaseguard --display-name "Python (PhaseGuard)"
```

### 2. Launch Jupyter Notebook
Launch the notebook server:
```bash
jupyter notebook
```

### 3. Select the Correct Kernel
* Create a new notebook or open `Layer1_Training.ipynb`.
* In the top-right corner, click **Kernel** -> **Change Kernel** -> Select **Python (PhaseGuard)**.
* You can now run PyTorch, Librosa, and train models directly from the notebook cells!

---

## 📡 The 5 Physical Security Signals (Explained)
Judges often ask how the detection parameters work. Here is the mathematical logic:

1. **Phase Jump Rate**: Counts frame-to-frame phase differences exceeding $\pi / 2$ radians. Human voice waves flow smoothly; frame-based AI vocoders exhibit sudden jumps at 20ms boundary transitions.
2. **Pitch Jitter**: Measures consecutive fundamental frequency ($f_0$) pitch variations. Real human voices wobble organically due to muscle tremors; AI voices either lack jitter entirely (static flat pitch) or display rigid mathematical tremor patterns.
3. **Spectral Flatness**: High flatness means white noise; low flatness means rich harmonics (speech formants). AI vocoders tend to blur formant details, yielding abnormally smooth spectral distributions.
4. **Noise Floor (RMS)**: Phone calls always have electrical/ambient noise. AI audio files are synthesized in near-perfect digital silence. We measure the energy (RMS) of the quietest 10% of frames to spot this dead digital background.
5. **MFCC Delta Variance**: Mel-Frequency Cepstral Coefficients represent vocal tract textures. Deltas represent the speed of vocal change. AI voices exhibit abnormally rigid transitions.

---

## 🚀 Execution & Verification Pipeline
Run these scripts in order to train the network and launch the system:

### 1. Generate Synthetic Voice Files
Synthesizes 200 real and 200 clone WAV files:
```bash
python generate_simulated_data.py
```

### 2. Process and Build the Dataset
Extracts signals and mel-spectrograms:
```bash
python build_dataset.py
```

### 3. Train Layer 1 Model
Trains the MobileNetV3 CNN model:
```bash
python train_layer1.py
```

### 4. Enroll Speaker Voiceprints
Loads the SpeechBrain ECAPA model, averages embeddings, and saves them:
```bash
python enroll_voices.py
```

### 5. Launch Dashboard
Run the Streamlit frontend:
```bash
python run_app.py
```
This opens the Web UI in your browser at `http://localhost:8501`.
>>>>>>> f910cc20bed285d385b74db0145c6f1ca13d6633
