import os
import shutil
import threading
import time
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from datetime import datetime, timedelta
from sqlalchemy import case, or_, text
from sqlalchemy.orm import Session
from typing import List, Optional

# Database & Config
from backend.app.config import settings
from backend.app.database import engine, Base, get_db
from backend.app.models.concept import ReferenceConcept
from backend.app.models.evaluation import Evaluation
from backend.app.schemas.concept import ConceptCreate, ConceptGenerate, ConceptResponse, TrendingResponse
from backend.app.schemas.evaluation import EvaluationResponse

# Services
from backend.app.services.speech_to_text import transcribe_audio
from backend.app.services.audio_analysis import analyze_audio_features
from backend.app.services.semantic import evaluate_semantic_understanding
from backend.app.services import discover, llm_client
from backend.app.services.evaluation import (
    FEEDBACK_DONE, FEEDBACK_FAILED, FEEDBACK_GENERATING, FEEDBACK_UNAVAILABLE,
    compute_fluency_score, evaluate_explanation, generate_llm_feedback,
)
from backend.app.services.pdf_report import generate_pdf_report

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="VBCUA API",
    description="Backend API for Voice-Based Concept Understanding Analyser",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Columns added after the first release; old rows get NULL (or the default given)
ADDED_EVALUATION_COLUMNS = {
    "final_score": "FLOAT DEFAULT 0.0",
    "asr_model": "VARCHAR(50)",
    "asr_confidence": "FLOAT",
    "pause_ratio_v1": "FLOAT",
    "cosine_similarity": "FLOAT",
    "keyword_coverage": "FLOAT",
    "mentioned_keywords": "JSON",
    "missing_keywords": "JSON",
    "understanding_score": "FLOAT",
    "semantic_method": "VARCHAR(10)",
    "fluency_score": "FLOAT",
    "fluency_method": "VARCHAR(10)",
    "feedback_source": "VARCHAR(20)",
    "llm_model": "VARCHAR(50)",
    "feedback_status": "VARCHAR(20)",
    "feedback_started_at": "TIMESTAMP",
    "feedback_seconds": "FLOAT",
    "asr_device": "VARCHAR(20)",
}

ADDED_CONCEPT_COLUMNS = {
    "source": "VARCHAR(20)",
}

DEFAULT_CONCEPT_TITLES = ("Machine Learning", "Cloud Computing", "Database Management System (DBMS)")


def migrate_evaluations_table(session_factory):
    """Dynamic database migration for existing tables (adds columns if missing)."""
    migrate_table(session_factory, "evaluations", ADDED_EVALUATION_COLUMNS)
    migrate_table(session_factory, "reference_concepts", ADDED_CONCEPT_COLUMNS)


def migrate_table(session_factory, table: str, columns: dict):
    for column, column_type in columns.items():
        db = session_factory()
        try:
            # Check if column exists by querying it
            db.execute(text(f"SELECT {column} FROM {table} LIMIT 1"))
        except Exception:
            db.rollback()
            logger.info(f"Database migration: {table}.{column} column missing. Appending it...")
            try:
                db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"))
                db.commit()
                logger.info(f"Database migration: {column} column successfully added.")
            except Exception as err:
                logger.error(f"Failed to add {column} column: {err}")
        finally:
            db.close()

def preload_models(loaders=None):
    """Loads Whisper and Sentence-BERT so the first attempt does not wait for them. Never raises."""
    if loaders is None:
        from backend.app.services.semantic import get_sbert_model
        from backend.app.services.speech_to_text import get_whisper_model
        loaders = {"Whisper": get_whisper_model, "Sentence-BERT": get_sbert_model}
    for name, load in loaders.items():
        start = time.time()
        try:
            load()
            logger.info(f"Preloaded {name} in {time.time() - start:.1f}s")
        except Exception as e:
            logger.warning(f"Could not preload {name}: {e}")

# Database Startup Actions
@app.on_event("startup")
def startup_db_setup():
    logger.info("Initializing database and tables...")
    Base.metadata.create_all(bind=engine)
    migrate_evaluations_table(lambda: next(get_db()))
    if settings.PRELOAD_MODELS:
        threading.Thread(target=preload_models, name="preload-models", daemon=True).start()

    # Seed default concepts if table is empty
    db = next(get_db())
    try:
        if db.query(ReferenceConcept).count() == 0:
            logger.info("Seeding default reference concepts...")
            default_concepts = [
                ReferenceConcept(
                    title="Machine Learning",
                    keywords=["supervised", "unsupervised", "data", "algorithms", "patterns", "predictions", "training"],
                    description="Machine Learning is a subset of artificial intelligence that focuses on building systems that learn from data. It involves training algorithms on large datasets so that the system can recognize patterns and make predictions or decisions without being explicitly programmed. The main types of machine learning are supervised learning, where models learn from labeled data, unsupervised learning, where models find hidden patterns in unlabeled data, and reinforcement learning, which uses a system of rewards and penalties."
                ),
                ReferenceConcept(
                    title="Cloud Computing",
                    keywords=["internet", "servers", "storage", "databases", "on-demand", "infrastructure", "scalability"],
                    description="Cloud Computing is the on-demand delivery of computing services over the internet, including servers, storage, databases, networking, software, and analytics. Instead of buying and maintaining physical data centers and servers, organizations can access technology services on an as-needed basis from cloud providers like AWS, Microsoft Azure, or Google Cloud. This provides flexibility, scalability, and cost savings, allowing businesses to scale their infrastructure dynamically."
                ),
                ReferenceConcept(
                    title="Database Management System (DBMS)",
                    keywords=["database", "sql", "query", "tables", "integrity", "transactions", "storage", "relational"],
                    description="A Database Management System (DBMS) is software designed to store, retrieve, define, and manage data in a database. It serves as an interface between the database and its end-users or programs, ensuring that data is consistently organized and easily accessible. A DBMS enforces data integrity constraints, manages database transactions, supports querying languages like SQL, and handles concurrent access control so multiple users can access the database safely without conflicts."
                )
            ]
            for concept in default_concepts:
                concept.source = "library"
            db.add_all(default_concepts)
            db.commit()
            logger.info("Seeding complete.")
        # The seeded concepts are the curated library (older databases stored no source)
        db.query(ReferenceConcept).filter(
            ReferenceConcept.source.is_(None), ReferenceConcept.title.in_(DEFAULT_CONCEPT_TITLES)
        ).update({"source": "library"}, synchronize_session=False)
        db.commit()
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()


# --- API Routes ---

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Voice-Based Concept Understanding Analyser API"}


# 1. Reference Concept Endpoints

@app.post("/api/concepts", response_model=ConceptResponse)
def create_concept(concept: ConceptCreate, db: Session = Depends(get_db)):
    db_concept = db.query(ReferenceConcept).filter(ReferenceConcept.title == concept.title).first()
    if db_concept:
        raise HTTPException(status_code=400, detail="Concept with this title already exists")
    
    new_concept = ReferenceConcept(
        title=concept.title,
        keywords=concept.keywords,
        description=concept.description,
        source="custom"
    )
    db.add(new_concept)
    db.commit()
    db.refresh(new_concept)
    return new_concept


@app.get("/api/concepts", response_model=List[ConceptResponse])
def list_concepts(db: Session = Depends(get_db)):
    """Curated library concepts first, then user-added ones, then AI-generated topics; by title within each."""
    group = case(
        (ReferenceConcept.source == "ai-generated", 2),
        (ReferenceConcept.source == "custom", 1),
        else_=0,
    )
    return db.query(ReferenceConcept).order_by(group, ReferenceConcept.title).all()


@app.get("/api/concepts/{concept_id}", response_model=ConceptResponse)
def get_concept(concept_id: int, db: Session = Depends(get_db)):
    concept = db.query(ReferenceConcept).filter(ReferenceConcept.id == concept_id).first()
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    return concept


# Discovery: trending topics and AI-written references for any topic

@app.get("/api/discover/trending", response_model=TrendingResponse)
def trending_topics(response: Response, db: Session = Depends(get_db)):
    """Today's trending topics (Gemini + Google Search, cached daily), or curated topics as a fallback."""
    # Never cached by the browser: curated topics change on every request
    response.headers["Cache-Control"] = "no-store"
    return discover.get_trending(db)


@app.post("/api/concepts/generate", response_model=ConceptResponse)
def generate_concept(request: ConceptGenerate, db: Session = Depends(get_db)):
    """Creates (or returns the existing) concept for any topic, with an AI-written reference and keywords."""
    try:
        return discover.generate_concept(db, request.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except llm_client.LLMNotConfigured:
        raise HTTPException(status_code=503, detail="AI is not configured (GEMINI_API_KEY is missing)")
    except llm_client.LLMError as e:
        logger.error(f"Reference generation failed for {request.topic!r}: {e}")
        if llm_client.is_quota_exhausted(e.__cause__ or e):
            raise HTTPException(status_code=429, detail=f"Gemini quota exhausted for {settings.GEMINI_DISCOVER_MODEL}: {e}")
        raise HTTPException(status_code=502, detail=f"Reference generation failed: {e}")


# 2. Evaluation Endpoints

@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate_audio(
    concept_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Retrieve Reference Concept
    concept = db.query(ReferenceConcept).filter(ReferenceConcept.id == concept_id).first()
    if not concept:
        raise HTTPException(status_code=404, detail="Reference concept not found")

    # Save Uploaded Audio File
    file_ext = os.path.splitext(file.filename)[1] or ".wav"
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    audio_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded audio file")

    # Evaluation Pipeline
    try:
        # Step A: Speech-to-Text Transcription
        asr_result = transcribe_audio(audio_path)
        transcription = asr_result["text"]
        if not transcription.strip():
            # If transcription is empty
            transcription = "[No speech detected or transcription failed]"
        
        logger.info(f"Audio transcription complete: {transcription[:100]}...")

        # Step B: Audio Signal Analysis
        audio_metrics = analyze_audio_features(audio_path, transcription)
        logger.info(f"Audio features extraction complete: {audio_metrics}")

        # Step C: Semantic Analysis
        semantic_metrics = evaluate_semantic_understanding(
            user_transcript=transcription,
            reference_description=concept.description,
            keywords=concept.keywords
        )
        logger.info(f"Semantic similarity calculated: {semantic_metrics['similarity_score']}%")

        # Step D: Fluency Analysis
        fluency_metrics = compute_fluency_score(
            filler_words_ratio=audio_metrics["filler_words_ratio"],
            pause_ratio=audio_metrics["pause_ratio_v1"]
        )
        logger.info(f"Fluency score calculated: {fluency_metrics['fluency_score']}/100")

        # Step E: Integrated Evaluation & Grading
        evaluation_result = evaluate_explanation(
            concept_title=concept.title,
            reference_description=concept.description,
            user_transcript=transcription,
            semantic_result=semantic_metrics,
            fluency_result=fluency_metrics
        )
        logger.info(f"Overall grade: {evaluation_result['understanding_grade']} ({evaluation_result['final_score']}%)")

        # Save to Database
        db_evaluation = Evaluation(
            concept_id=concept_id,
            transcription=transcription,
            asr_model=asr_result["asr_model"],
            asr_confidence=asr_result["asr_confidence"],
            asr_device=asr_result.get("asr_device"),
            semantic_score=semantic_metrics["similarity_score"],
            cosine_similarity=semantic_metrics["cosine_similarity"],
            keyword_coverage=semantic_metrics["keyword_coverage"],
            mentioned_keywords=semantic_metrics["mentioned_keywords"],
            missing_keywords=semantic_metrics["missing_keywords"],
            understanding_score=semantic_metrics["combined_score"],
            semantic_method=semantic_metrics["scoring_method"],
            fluency_score=fluency_metrics["fluency_score"],
            fluency_method=fluency_metrics["scoring_method"],
            filler_words_count=audio_metrics["filler_words_count"],
            filler_words_ratio=audio_metrics["filler_words_ratio"],
            pause_ratio=audio_metrics["pause_ratio"],
            pause_ratio_v1=audio_metrics["pause_ratio_v1"],
            rms_energy=audio_metrics["rms_energy"],
            speech_duration=audio_metrics["duration"],
            final_score=evaluation_result["final_score"],
            understanding_grade=evaluation_result["understanding_grade"],
            feedback=evaluation_result["feedback"],
            feedback_source=evaluation_result["feedback_source"],
            llm_model=evaluation_result["llm_model"],
            feedback_status=evaluation_result["feedback_status"],
            audio_path=audio_path
        )
        db.add(db_evaluation)
        db.commit()
        db.refresh(db_evaluation)
        
        # Populate relationships for Pydantic response serialization
        # Just loading the concept relationship
        _ = db_evaluation.concept
        
        return db_evaluation

    except Exception as e:
        logger.error(f"Error executing evaluation pipeline: {e}", exc_info=True)
        # Clean up audio file on failure
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up file {audio_path}: {cleanup_err}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/evaluations", response_model=List[EvaluationResponse])
def list_evaluations(db: Session = Depends(get_db)):
    return db.query(Evaluation).order_by(Evaluation.created_at.desc()).all()


@app.get("/api/evaluations/{eval_id}", response_model=EvaluationResponse)
def get_evaluation(eval_id: int, db: Session = Depends(get_db)):
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


# A "generating" claim older than this is treated as abandoned (e.g. the server restarted)
STALE_FEEDBACK_SECONDS = 300


@app.post("/api/evaluations/{eval_id}/feedback", response_model=EvaluationResponse)
def generate_feedback(eval_id: int, db: Session = Depends(get_db)):
    """
    Generates AI feedback for a stored attempt and replaces the rule-based feedback.
    Safe to call repeatedly: if the attempt already has AI feedback, or another request
    is generating it, the current state is returned without calling Gemini again.
    Scores and grade are never changed.
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    if evaluation.feedback_source == "llm":
        return evaluation
    if not llm_client.is_configured():
        evaluation.feedback_status = FEEDBACK_UNAVAILABLE
        db.commit()
        db.refresh(evaluation)
        return evaluation

    # Atomically claim the attempt so two concurrent requests cannot both call Gemini
    now = datetime.utcnow()
    claimed = db.query(Evaluation).filter(
        Evaluation.id == eval_id,
        or_(Evaluation.feedback_source.is_(None), Evaluation.feedback_source != "llm"),
        or_(
            Evaluation.feedback_status.is_(None),
            Evaluation.feedback_status != FEEDBACK_GENERATING,
            Evaluation.feedback_started_at.is_(None),
            Evaluation.feedback_started_at < now - timedelta(seconds=STALE_FEEDBACK_SECONDS),
        ),
    ).update({"feedback_status": FEEDBACK_GENERATING, "feedback_started_at": now}, synchronize_session=False)
    db.commit()
    db.refresh(evaluation)
    if not claimed:
        return evaluation

    semantic_result = {
        "similarity_score": evaluation.semantic_score,
        "keyword_coverage": evaluation.keyword_coverage,
        "mentioned_keywords": evaluation.mentioned_keywords or [],
        "missing_keywords": evaluation.missing_keywords or [],
        "combined_score": evaluation.understanding_score,
    }
    pause_ratio = evaluation.pause_ratio_v1 if evaluation.pause_ratio_v1 is not None else evaluation.pause_ratio
    fluency_result = compute_fluency_score(evaluation.filler_words_ratio or 0.0, pause_ratio or 0.0)
    try:
        result = generate_llm_feedback(
            concept_title=evaluation.concept.title,
            reference_description=evaluation.concept.description,
            user_transcript=evaluation.transcription,
            semantic_result=semantic_result,
            fluency_result=fluency_result,
        )
        evaluation.feedback = result["feedback"]
        evaluation.feedback_source = "llm"
        evaluation.llm_model = result["llm_model"]
        evaluation.feedback_seconds = result["feedback_seconds"]
        evaluation.feedback_status = FEEDBACK_DONE
        logger.info(f"AI feedback for attempt {eval_id} from {result['llm_model']} in {result['feedback_seconds']}s")
    except Exception as e:
        evaluation.feedback_status = FEEDBACK_FAILED
        evaluation.feedback_seconds = getattr(e, "feedback_seconds", None)
        logger.error(f"AI feedback for attempt {eval_id} failed; rule-based feedback kept: {e}")
    db.commit()
    db.refresh(evaluation)
    _ = evaluation.concept
    return evaluation


@app.get("/api/evaluations/{eval_id}/pdf")
def download_pdf(eval_id: int, db: Session = Depends(get_db)):
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    concept = db.query(ReferenceConcept).filter(ReferenceConcept.id == evaluation.concept_id).first()
    concept_title = concept.title if concept else "Concept"
    
    # Generate unique PDF filename
    pdf_filename = f"report_{eval_id}_{uuid.uuid4().hex[:6]}.pdf"
    pdf_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
    
    try:
        generate_pdf_report(
            evaluation=evaluation,
            concept_title=concept_title,
            output_path=pdf_path
        )
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed")
            
        return FileResponse(
            path=pdf_path,
            filename=f"VBCUA_Report_{concept_title.replace(' ', '_')}_{eval_id}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating or downloading PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


# ---------- Web frontend (React build in web/dist) ----------
# Registered last so every /api route above takes precedence.

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def resolve_web_file(dist: Path, request_path: str) -> Optional[Path]:
    """
    The file to serve for a browser path: an existing file inside dist, else index.html
    (client-side routing). Returns None if the app is not built. Never leaves dist.
    """
    index = dist / "index.html"
    if not index.is_file():
        return None
    candidate = (dist / request_path).resolve()
    if request_path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
        return candidate
    return index


@app.get("/{full_path:path}", include_in_schema=False)
def serve_web(full_path: str):
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    target = resolve_web_file(WEB_DIST, full_path)
    if target is None:
        raise HTTPException(status_code=404, detail="Web app is not built. Run: cd web && npm install && npm run build")
    # Vite's hashed files in assets/ never change; everything else is revalidated
    immutable = full_path.startswith("assets/") and target.name != "index.html"
    headers = {"Cache-Control": "public, max-age=31536000, immutable" if immutable else "no-cache"}
    return FileResponse(target, headers=headers)
