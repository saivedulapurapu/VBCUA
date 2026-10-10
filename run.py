import os
import sys
import subprocess
import time
import shutil
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_env():
    """
    Checks if a .env file exists; if not, copies it from the template.
    """
    if not os.path.exists(".env"):
        logger.info("No .env file found. Copying from .env.template...")
        if os.path.exists(".env.template"):
            shutil.copy(".env.template", ".env")
            logger.info(".env created from template. Please set GEMINI_API_KEY inside if desired.")
        else:
            with open(".env", "w") as f:
                f.write("DATABASE_URL=sqlite:///./vbcua.db\n")
                f.write("GEMINI_API_KEY=\n")
                f.write("WHISPER_MODEL=small\n")
                f.write("UPLOAD_DIR=./uploads\n")
            logger.info("Created a basic .env file.")
    else:
        logger.info(".env file is present.")

def setup_db():
    """
    For a PostgreSQL URL, creates the target db if it does not exist.
    Skipped for SQLite (the default) and any other non-PostgreSQL URL.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "sqlite:///./vbcua.db")

    if not db_url.startswith("postgresql"):
        logger.info("Non-PostgreSQL database configured; skipping PostgreSQL setup. Tables are created by FastAPI.")
        return
        
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        result = urlparse(db_url)
        username = result.username or "postgres"
        password = result.password or "postgres"
        hostname = result.hostname or "localhost"
        port = result.port or 5432
        dbname = result.path.lstrip('/') or "vbcua_db"
        
        logger.info(f"Checking if target database '{dbname}' exists on PostgreSQL at {hostname}:{port}...")
        
        # Connect to default database 'postgres' to run the verification/creation query
        conn = psycopg2.connect(
            dbname="postgres",
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{dbname}'")
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Database '{dbname}' does not exist. Creating it now...")
            cursor.execute(f"CREATE DATABASE {dbname}")
            logger.info(f"Database '{dbname}' created successfully.")
        else:
            logger.info(f"Database '{dbname}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not verify/create PostgreSQL database automatically: {e}.")
        logger.warning("The application will fallback to SQLite if PostgreSQL connection fails at runtime.")

def start_services():
    """
    Launches FastAPI (uvicorn) and Streamlit dashboard under the virtual environment.
    """
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    venv_streamlit = os.path.join(".venv", "Scripts", "streamlit.exe")
    
    if not os.path.exists(venv_python):
        logger.warning("Virtual environment not found. Running with system python/streamlit.")
        venv_python = "python"
        venv_streamlit = "streamlit"
        
    logger.info("Starting FastAPI backend...")
    
    # Start FastAPI
    backend_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    
    # Allow 2 seconds to bind port
    time.sleep(2)
    
    if backend_proc.poll() is not None:
        logger.error("FastAPI backend failed to start. Please check the terminal logs above.")
        return
        
    logger.info("FastAPI backend successfully launched at http://127.0.0.1:8000")
    logger.info("Starting Streamlit frontend dashboard...")
    
    # Start Streamlit
    # --server.port 8501 is default, but explicitly setting it avoids conflicts
    frontend_proc = subprocess.Popen(
        [venv_streamlit, "run", "frontend/app.py", "--server.port", "8501"]
    )
    
    logger.info("Streamlit dashboard successfully launched at http://localhost:8501")
    logger.info("Application is running. Press Ctrl+C to terminate both servers.")
    
    try:
        while True:
            # Monitor processes
            if backend_proc.poll() is not None:
                logger.error("FastAPI backend process crashed.")
                break
            if frontend_proc.poll() is not None:
                logger.error("Streamlit frontend process crashed.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Termination signal received. Stopping services...")
    finally:
        # Graceful shutdown
        backend_proc.terminate()
        frontend_proc.terminate()
        
        try:
            backend_proc.wait(timeout=5)
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
            frontend_proc.kill()
            
        logger.info("Services stopped. Exiting.")

if __name__ == "__main__":
    check_env()
    setup_db()
    start_services()
