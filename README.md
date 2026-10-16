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
   - Fluency score (provisional) is based on the pause ratio only, measured between the first and last detected speech using **Librosa**; calibration is pending.
   - Delivery measures are stored per attempt but **not scored**: speaking rate (words per minute), number of pauses of 1 s or more, long pauses per minute and the longest pause.
   - Filler words (`um`, `uh`, `like`, `ah`, `you know`, `basically`, `actually`) are counted and stored but are **experimental and not scored**: speech-to-text usually removes "um/uh".
   - RMS energy is stored as a raw value and used only to warn about a very quiet recording ("check your microphone"); it does not affect any score.
   - Overall score = 70% understanding + 30% fluency.
   - The original formulas remain available as baseline `B0` (`method="b0"`) for comparison.
   - Every sub-score, the scoring methods, the ASR model and the feedback source (LLM or rule-based) are stored per attempt.

3. **Web app** (React + TypeScript + Vite, served by the backend)
   - Glassmorphism design in light and dark, following the system setting unless you pick one.
   - **Trending today**: topics in today's news (any area, not only tech) found by Gemini with Google Search grounding, cached once per day, with source links and Google's search suggestions. If grounding is unavailable (no key or quota), a curated list is shown as "Suggested topics".
   - **Any topic**: type anything; Gemini writes a reference definition and key terms, saved as a concept marked *AI-generated reference* (for practice; the `source` column records where each concept came from).
   - Three steps: choose a topic (searchable, or **Surprise me** for a random one), record in the browser with a live waveform or upload a file, review the results. The selected concept and the open attempt are kept in the URL, so a page reload keeps them.
   - Scores appear as soon as the analysis finishes; AI feedback follows a few seconds later ("Writing feedback…").
   - Results: overall, understanding and fluency score rings with short explanations, and tabs for feedback, keywords, transcript and every stored sub-score.
   - History; selecting an attempt opens it instantly.
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
├── web/                       # React + TypeScript web app (Vite)
│   ├── src/                   # Components, hooks, API client, styles and tests
│   └── package.json           # Web app dependencies and scripts
├── tests/                     # pytest suite (models and APIs stubbed)
├── uploads/                   # Local store for audio uploads and generated PDFs
├── VBCUA_Documents/           # June 2026 project documents
├── .env                       # Local environment configuration file (ignored)
├── .env.template              # Template for configuring env variables
├── pytest.ini                 # pytest configuration
├── run.py                     # Builds the web app if needed and starts the server
└── README.md                  # Project documentation
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10–3.12** (tested with 3.11; PyTorch/Whisper are not tested on 3.13 here)
- **Node.js 18+** (with npm): builds the web app. `run.py` builds it automatically.
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

# Gemini model for trending topics and AI-written references (uses Google Search grounding)
GEMINI_DISCOVER_MODEL=gemini-3.5-flash-lite

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
```

---

## Running the Application

`run.py` checks your `.env`, creates the PostgreSQL database if a `postgresql` URL is configured (skipped for SQLite), builds the web app when its sources have changed (installing npm packages on the first run), and starts the server. Missing columns are added to existing databases at startup.

Run the launcher:
```bash
python run.py
```

- **App**: `http://127.0.0.1:8000`
- **API documentation**: `http://127.0.0.1:8000/docs`

To work on the web app with hot reload, run the backend (`python run.py`) and, in a second terminal, `cd web && npm run dev` (opens `http://localhost:5173`; API calls are proxied to the backend).

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/concepts` | Create a concept |
| GET | `/api/concepts` | List concepts |
| POST | `/api/concepts/generate` | Create a concept for any topic with an AI-written reference |
| GET | `/api/concepts/{concept_id}` | Get one concept |
| GET | `/api/discover/trending` | Today's trending topics (Google Search via Gemini, cached daily) or curated topics |
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
python -m pytest          # backend
cd web && npm test        # web app (Vitest + Testing Library)
```

The tests stub Whisper, Sentence-BERT and Gemini, so they need no model downloads, API key or network access.

---

## Pipeline & Tech Stack

- **Frameworks**: FastAPI (backend) & React + TypeScript + Vite (web app)
- **Database**: SQLite by default; PostgreSQL optional (`postgresql+psycopg2://` URL) via SQLAlchemy ORM
- **Speech processing**: OpenAI Whisper (optional Google Gemini fallback)
- **AI feedback**: Google Gemini via the `google-genai` SDK, with a rule-based fallback
- **NLP / Semantics**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Audio DSP**: Librosa & NumPy
- **Report Generation**: ReportLab PDF library
