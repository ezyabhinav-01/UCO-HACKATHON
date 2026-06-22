# 🛡️ PhaseGuard
**Real-Time Two-Layer Voice Forensics for Banking Security & Deepfake Interception**  
*UCO Bank Hackathon 2026 — Built by Team Ozymandias*

---

## 📌 Project Overview
PhaseGuard is an advanced, real-time voice verification system designed to protect banking transactions against AI voice cloning fraud. If a criminal uses a cloned voice (e.g., generated via ElevenLabs) to call a bank, PhaseGuard intercepts and blocks the call in real-time before any funds are moved.

It operates using two sequential layers:
*   **Layer 1 (AI Voice Authenticity):** Detects neural vocoder phase artifacts and boundary jumps using physical signal checks and a fine-tuned MobileNetV3 CNN.
*   **Layer 2 (Speaker Verification):** Verifies that the speaker's vocal signature matches the enrolled customer using ECAPA-TDNN embeddings compared against stored customer voiceprints in PostgreSQL (via pgvector).

---

## 🏗️ Architecture

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
    → identity decision (threshold: 0.65)
    → Risk Engine (combines Layer 1 + Layer 2 scores)
    → persist verification_log + risk_log
    → return VerificationResult
```

### Risk Engine Rules

| Condition | Decision |
|---|---|
| `layer1_score > 0.70` | `FRAUD_ALERT` |
| `speaker_similarity < 0.65` | `FRAUD_ALERT` |
| otherwise | `CLEAN` |

---

## 📂 Project Structure

```text
E:/UCO-HACKATHON/
├── app/                         # Layer 2 FastAPI Backend
│   ├── api/v1/endpoints/        # API route handlers (enroll, verify, users, history)
│   ├── core/                    # App configuration (Pydantic settings) and logging
│   ├── database/                # Database engine, session, and async dependencies
│   ├── models/                  # SQLAlchemy models (users, voiceprints, logs)
│   ├── repositories/            # Database abstraction layer
│   ├── services/                # Business logic orchestration
│   ├── ml/                      # SpeechBrain ECAPA-TDNN inference wrapper
│   ├── utils/                   # Audio I/O and exception handlers
│   └── main.py                  # FastAPI server entry point
├── streamlit_app/               # Layer 2 Streamlit Web Frontend
│   ├── app.py                   # Multi-page dashboard entry
│   ├── api_client.py            # API request layer
│   ├── config.py                # Frontend thresholds and endpoints
│   └── pages/                   # Enrollment, Verification, History, Risk dashboards
├── data/                        # Audio Datasets
│   ├── real_voices/             # Genuine speaker recordings (16kHz WAV mono)
│   ├── clones/                  # Deepfaked voice clone clips (16kHz WAV mono)
│   ├── sources/                 # External datasets (LJSpeech, Common Voice, Svarah)
│   └── processed/               # Compiled datasets and train/test splits (.npy matrices)
├── models/                      # Saved PyTorch Weights & Local Templates
│   ├── layer1_mobilenet.pth     # Trained MobileNetV3 CNN weights
│   └── voiceprints.pth          # Backup of enrolled ECAPA-TDNN speaker templates
├── pretrained_models/           # Cached SpeechBrain ECAPA models (downloaded automatically)
├── alembic/                     # Database migrations
├── tests/                       # Pytest unit and integration test suite
├── Dockerfile                   # FastAPI backend image
├── Dockerfile.streamlit         # Streamlit frontend image
├── docker-compose.yml           # Database + API + Streamlit deployment configuration
├── requirements.txt             # Python environment dependencies
├── generate_simulated_data.py   # Synthesizes mock audio clips for testing
├── preprocess.py                # Audio normalizer and 5-signal physical feature extractor
├── build_dataset.py             # Compiles raw WAV files into unified numpy datasets
├── process_phase1.py            # Processes and splits multi-source, speaker-aware data
├── train_layer1.py              # PyTorch script training the MobileNetV3 CNN model
├── enroll_voices.py             # SpeechBrain script building local voiceprints
├── dashboard.py                 # Integrated local Streamlit dashboard
├── run_app.py                   # Launcher script for the local dashboard
├── SETUP.md                     # Rapid teammate setup guide
└── README.md                    # Core project documentation
```

---

## 🛠️ Step-by-Step Environment Setup

Teammates can reproduce the exact working environment using a clean python virtual environment or Anaconda.

### Local Python Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Anaconda Conda Setup (Alternative)

```bash
# Create dedicated virtual environment with Python 3.10
conda create -n phaseguard python=3.10 -y
conda activate phaseguard

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Pipelines

You can run PhaseGuard in two modes: **Local Integrated Mode** (quick training & local evaluation) or **Database-Backed Production Mode** (FastAPI backend + PostgreSQL + Multi-page Web UI).

### Option A: Local Integrated Mode (Notebooks & Dashboard)

Run these scripts in order to train the Layer 1 CNN, enroll voiceprints locally, and start the integrated dashboard:

1.  **Generate Synthetic Voice Files** (Synthesizes 200 real and 200 clone WAV files):
    ```bash
    python generate_simulated_data.py
    ```
2.  **Process and Build the Dataset** (Extracts features, spectrograms, and applies splits):
    ```bash
    python build_dataset.py
    ```
    *Or run `process_phase1.py` for multi-source, speaker-aware splits.*
3.  **Train the Layer 1 Model**:
    ```bash
    python train_layer1.py
    ```
4.  **Enroll Speaker Voiceprints**:
    ```bash
    python enroll_voices.py
    ```
5.  **Launch the Integrated Local Dashboard**:
    ```bash
    python run_app.py
    ```
    *This runs the local `dashboard.py` frontend on `http://localhost:8501`.*

### Option B: Database-Backed Production Mode (FastAPI + Streamlit + Postgres)

1.  **Configure Environment Variables**:
    ```bash
    copy .env.example .env
    ```
    Edit `.env` to supply the database URL connecting to your Postgres / Neon instance.
2.  **Run Database Migrations**:
    ```bash
    alembic upgrade head
    ```
3.  **Start the Backend API**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
4.  **Start the Production Web App** (Open a new terminal):
    ```bash
    cd streamlit_app
    streamlit run app.py
    ```
    *This opens the multi-page portal on `http://localhost:8501`.*

---

## 📡 The 5 Physical Security Signals (Explained)

Judges or users interested in how PhaseGuard catches synthetic/cloned speech can refer to these acoustic parameters:

1.  **Phase Jump Rate**: Counts frame-to-frame phase differences exceeding $\pi / 2$ radians. Human voice waves flow smoothly; frame-based AI vocoders exhibit sudden jumps at 20ms boundary transitions.
2.  **Pitch Jitter**: Measures consecutive fundamental frequency ($f_0$) pitch variations. Real human voices wobble organically due to muscle tremors; AI voices either lack jitter entirely (static flat pitch) or display rigid mathematical tremor patterns.
3.  **Spectral Flatness**: High flatness means white noise; low flatness means rich harmonics (speech formants). AI vocoders tend to blur formant details, yielding abnormally smooth spectral distributions.
4.  **Noise Floor (RMS)**: Phone calls always have electrical/ambient noise. AI audio files are synthesized in near-perfect digital silence. We measure the energy (RMS) of the quietest 10% of frames to spot this dead digital background.
5.  **MFCC Delta Variance**: Mel-Frequency Cepstral Coefficients represent vocal tract textures. Deltas represent the speed of vocal change. AI voices exhibit abnormally rigid transitions.

---

## 🧪 Running Tests

Ensure your changes do not break existing functionality by running the test suite:
```bash
pytest tests/ -v
```

All 45 tests should pass successfully.

---

## 📊 Database Schema

For PostgreSQL with the `pgvector` extension enabled, the database schema is structured as follows:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

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
