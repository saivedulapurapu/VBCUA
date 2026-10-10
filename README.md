# Voice-Based Concept Understanding Analyser (VBCUA)

VBCUA is an AI-powered web application designed to evaluate how effectively users understand and explain conceptual topics through spoken communication. The platform combines speech-to-text transcription, semantic similarity analysis, audio feature extraction, and intelligent scoring mechanisms to assess both the quality of conceptual understanding and the fluency of speech delivery.

## Demo & Project Links
- Project Demonstration Video: [Google Drive](https://drive.google.com/file/d/1en8HLZQfWnESEw4s4gTvV9vfchR6EO77/view?usp=drivesdk)
- Code Demonstration Video: [Google Drive](https://drive.google.com/file/d/1V5h2Jw1n2WLrSlF-aw7fje6Yl8zbspC7/view?usp=drivesdk)

---

## Key Features

1. **Semantic Concept Evaluation**
   - Transcribes audio explanations using local **OpenAI Whisper** (`small` by default). An optional **Google Gemini** fallback is used only if Whisper fails and `STT_GEMINI_FALLBACK=true`. The ASR model used is saved with every attempt.
   - Compares the explanation to predefined reference definitions using **Sentence-BERT (S-BERT)** semantic embeddings (similarity = `max(0, cosine) × 100`).
   - Checks keyword coverage with whole-word matching (case-insensitive; hyphen/space and simple plurals treated as equal; no substring matches).
   - Understanding score = 70% similarity + 30% keyword coverage.
   - Assigns qualitative grades: *Strong Understanding* (≥ 80), *Moderate Understanding* (≥ 50), or *Poor Understanding*. The cut-offs are not yet calibrated against human ratings.

2. **Speech Fluency & Delivery Metrics**
   - Fluency score is based on the pause ratio, measured between the first and last detected speech using **Librosa**.
   - Filler words (`um`, `uh`, `like`, `ah`, `you know`, `basically`, `actually`) are counted and stored but are **experimental and not scored**: speech-to-text usually removes "um/uh".
   - RMS energy is stored as a raw value and used only to warn about a very quiet recording ("check your microphone"); it does not affect any score.
   - Overall score = 70% understanding + 30% fluency.
   - The original formulas remain available as baseline `B0` (`method="b0"`) for research comparisons.
   - Every sub-score, the scoring methods, the ASR model and the feedback source (LLM or rule-based) are stored per attempt.

3. **Dashboard**
   - Three steps: choose a concept, record or upload an explanation, review the results. The selected concept and the open attempt are kept in the URL, so a page reload keeps them.
   - Live progress while an attempt is analysed; clear error messages with technical details on demand.
   - Results: overall, understanding and fluency scores with short explanations, and tabs for feedback, keywords, transcript and every stored sub-score.
   - History table; selecting a row opens that attempt.
   - Downloads assessment reports as **ReportLab PDFs**.

There are no user accounts or login: all attempts are stored in one shared history. Login and registration appear in the June 2026 project documents as planned features; see [Scope and changes since June 2026](VBCUA_Documents/Scope_and_Changes.md).

---

## Project Structure

```text
VBCUA/
├── backend/
│   ├── app/
│   │   ├── config.py          # Settings and environment configuration
│   │   ├── database.py        # SQLAlchemy engine and connection manager
│   │   ├── main.py            # FastAPI main application and endpoints
│   │   ├── models/            # SQLAlchemy database models (ReferenceConcept, Evaluation)
│   │   ├── schemas/           # Pydantic serialization models
│   │   └── services/          # AI & DSP Pipeline
│   │       ├── audio_analysis.py   # Librosa signal processing & filler word counts
│   │       ├── evaluation.py       # Fluency calculations & AI tutor feedback
│   │       ├── llm_client.py       # Single Gemini (google-genai) client for all LLM calls
│   │       ├── pdf_report.py       # ReportLab PDF generator
│   │       ├── semantic.py         # Sentence-BERT similarity & keyword check
│   │       └── speech_to_text.py   # Whisper & Gemini transcription managers
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── app.py                 # Streamlit frontend dashboard
│   ├── api.py                 # Backend API calls with user-facing error messages
│   ├── display.py             # Display helpers (stored scores, HTML escaping)
│   ├── requirements.txt       # Frontend dependencies
│   └── styles.css             # Small style additions to the light theme
├── .streamlit/config.toml     # Streamlit theme (light) and toolbar settings
├── tests/                     # pytest suite (models and APIs stubbed)
├── research/data/             # Research data (git-ignored)
├── uploads/                   # Local store for audio uploads and generated PDFs
├── VBCUA_Documents/           # June 2026 project documents and research plan
├── .env                       # Local environment configuration file (ignored)
├── .env.template              # Template for configuring env variables
├── pytest.ini                 # pytest configuration
├── run.py                     # Orchestration script to run backend and frontend
└── README.md                  # Project documentation
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10–3.12** (tested with 3.11; PyTorch/Whisper are not tested on 3.13 here)
- **FFmpeg**: Required by Librosa/Whisper. 
  * *Windows*: Install FFmpeg (e.g. `winget install --id Gyan.FFmpeg -e`) and make sure `ffmpeg -version` works in a new terminal.

### Step 1: Clone the Repository
```bash
git clone https://github.com/saivedulapurapu/VBCUA.git
cd VBCUA
```

### Step 2: Configure Environment Variables
Copy the template `.env.template` file to `.env`:
```bash
cp .env.template .env
```
Inside `.env`, configure your settings:
```ini
# Database: SQLite (local vbcua.db file) by default
DATABASE_URL=sqlite:///./vbcua.db
# Optional PostgreSQL (uses the psycopg2 driver from requirements):
# DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/vbcua_db

# Google Gemini API Key (Required for advanced AI feedback summaries)
GEMINI_API_KEY=gemini_api_key_here

# Gemini model for all LLM calls (see https://ai.google.dev/gemini-api/docs/models)
GEMINI_MODEL=gemini-3.8-flash

# Gemini model for written feedback (faster; GEMINI_MODEL is used for rubric grading)
GEMINI_FEEDBACK_MODEL=gemini-3.5-flash-lite

# Gemini thinking effort for feedback (minimal, low, medium, high; empty = model default)
GEMINI_THINKING_LEVEL=low

# Load Whisper and Sentence-BERT at backend startup
PRELOAD_MODELS=true

# Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=small

# Device for Whisper and Sentence-BERT: cpu, cuda or auto (cuda needs a CUDA build of PyTorch)
DEVICE=cpu

# Fall back to Gemini transcription if local Whisper fails
STT_GEMINI_FALLBACK=false
```

### Step 3: Install Dependencies
It is highly recommended to use a Python virtual environment:
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

---

## Running the Application

VBCUA comes with an orchestrator script `run.py` that will verify your `.env` configuration, create the PostgreSQL database if a `postgresql` URL is configured (skipped for SQLite), and start both services in parallel. Missing columns are added to existing databases at backend startup.

Run the launcher:
```bash
python run.py
```

- **FastAPI Backend**: Runs at `http://127.0.0.1:8000` (API documentation at `/docs`).
- **Streamlit Frontend**: Opens automatically at `http://localhost:8501`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/concepts` | Create a concept |
| GET | `/api/concepts` | List concepts |
| GET | `/api/concepts/{concept_id}` | Get one concept |
| POST | `/api/evaluate` | Upload audio and run the evaluation pipeline |
| GET | `/api/evaluations` | List past attempts |
| GET | `/api/evaluations/{eval_id}` | Get one attempt |
| POST | `/api/evaluations/{eval_id}/feedback` | Generate AI feedback for an attempt (safe to call twice) |
| GET | `/api/evaluations/{eval_id}/pdf` | Download the PDF report for an attempt |

Interactive documentation is served at `/docs` while the backend is running.

---

## Running Tests

```bash
pip install pytest
python -m pytest
```

The tests stub Whisper, Sentence-BERT and Gemini, so they need no model downloads, API key or network access.

---

## Pipeline & Tech Stack

- **Frameworks**: FastAPI (Backend) & Streamlit (Frontend)
- **Database**: SQLite by default; PostgreSQL optional (`postgresql+psycopg2://` URL) via SQLAlchemy ORM
- **Speech processing**: OpenAI Whisper (optional Google Gemini fallback)
- **AI feedback**: Google Gemini via the `google-genai` SDK, with a rule-based fallback
- **NLP / Semantics**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Audio DSP**: Librosa & NumPy
- **Report Generation**: ReportLab PDF library
