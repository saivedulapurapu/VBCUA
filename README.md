# Voice-Based Concept Understanding Analyser (VBCUA)

VBCUA is an AI-powered web application designed to evaluate how effectively users understand and explain conceptual topics through spoken communication. The platform combines speech-to-text transcription, semantic similarity analysis, audio feature extraction, and intelligent scoring mechanisms to assess both the quality of conceptual understanding and the fluency of speech delivery.

## Demo & Project Links
- Project Demonstration Video: [Google Drive](https://drive.google.com/file/d/1en8HLZQfWnESEw4s4gTvV9vfchR6EO77/view?usp=drivesdk)
- Code Demonstration Video: [Google Drive](https://drive.google.com/file/d/1V5h2Jw1n2WLrSlF-aw7fje6Yl8zbspC7/view?usp=drivesdk)

---

## Key Features

1. **Semantic Concept Evaluation**
   - Transcribes audio explanations using local **OpenAI Whisper** (`small` by default). An optional **Google Gemini** fallback is used only if Whisper fails and `STT_GEMINI_FALLBACK=true`. The ASR model used is saved with every attempt.
   - Compares the explanation to predefined reference definitions using **Sentence-BERT (S-BERT)** semantic embeddings.
   - Evaluates keyword coverage against essential terms.
   - Assigns qualitative grades: *Strong Understanding*, *Moderate Understanding*, or *Poor Understanding*.

2. **Speech Fluency & Delivery Metrics**
   - Analyzes filler word usage (`um`, `uh`, `like`, `ah`, `you know`, `basically`, `actually`).
   - Extracts pause-to-speech ratio and silence thresholds using **Librosa** audio signal processing.
   - Measures speaking volume intensity using Root Mean Square (RMS) energy.
   - Computes a weighted overall score ($70\%$ semantic comprehension + $30\%$ speech fluency).

3. **Interactive Reporting Dashboard**
   - Displays real-time dark-cyber style waveform visualization of user recordings.
   - Interactive gauge displays for scores and color-coded keyword badges.
   - Historical evaluation timeline.
   - Downloads styled assessment reports as **ReportLab PDFs**.

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
│   │       ├── pdf_report.py       # ReportLab PDF generator
│   │       ├── semantic.py         # Sentence-BERT similarity & keyword check
│   │       └── speech_to_text.py   # Whisper & Gemini transcription managers
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── app.py                 # Streamlit frontend dashboard
│   ├── requirements.txt       # Frontend dependencies
│   └── styles.css             # Premium glassmorphism dark-cyber CSS styles
├── uploads/                   # Local store for audio uploads and generated PDFs
├── .env                       # Local environment configuration file (ignored)
├── .env.template              # Template for configuring env variables
├── run.py                     # Orchestration script to run backend and frontend
└── README.md                  # Project documentation
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10+**
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
# Database configuration (Defaults to PostgreSQL, falls back to SQLite vbcua.db)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vbcua_db

# Google Gemini API Key (Required for advanced AI feedback summaries)
GEMINI_API_KEY=gemini_api_key_here

# Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=small

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

VBCUA comes with an orchestrator script `run.py` that will verify your `.env` configuration, ensure your target database exists (with self-healing migrations), and start both services in parallel.

Run the launcher:
```bash
python run.py
```

- **FastAPI Backend**: Runs at `http://127.0.0.1:8000` (API documentation at `/docs`).
- **Streamlit Frontend**: Opens automatically at `http://localhost:8501`.

---

## Pipeline & Tech Stack

- **Frameworks**: FastAPI (Backend) & Streamlit (Frontend)
- **Database**: PostgreSQL (Production) / SQLite (Fallback) via SQLAlchemy ORM
- **Speech processing**: OpenAI Whisper & Google Gemini
- **NLP / Semantics**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Audio DSP**: Librosa & NumPy
- **Report Generation**: ReportLab PDF library
