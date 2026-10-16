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

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
WEB_SOURCES = ("src", "public", "index.html", "package.json", "vite.config.js")


def newest_mtime(paths):
    """Latest modification time of any file under the given paths (0 if none exist)."""
    newest = 0.0
    for path in paths:
        if os.path.isfile(path):
            newest = max(newest, os.path.getmtime(path))
        for root, _dirs, files in os.walk(path):
            for name in files:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
    return newest


def web_needs_build(web_dir=None):
    """True if web/dist is missing or older than any frontend source file."""
    web_dir = web_dir or WEB_DIR
    index = os.path.join(web_dir, "dist", "index.html")
    if not os.path.exists(index):
        return True
    return newest_mtime(os.path.join(web_dir, p) for p in WEB_SOURCES) > os.path.getmtime(index)


def build_web():
    """
    Builds the React app into web/dist when needed (installs packages on first run).
    Returns False if it cannot be built and no previous build exists.
    """
    if not web_needs_build():
        logger.info("Web app is up to date.")
        return True
    npm = shutil.which("npm")
    has_build = os.path.exists(os.path.join(WEB_DIR, "dist", "index.html"))
    if npm is None:
        if has_build:
            logger.warning("Node.js/npm not found; serving the existing (possibly outdated) web build.")
            return True
        logger.error("Node.js 18+ is required to build the web app (https://nodejs.org). Then run: python run.py")
        return False
    try:
        if not os.path.isdir(os.path.join(WEB_DIR, "node_modules")):
            logger.info("Installing web app packages (first run)...")
            subprocess.run([npm, "ci"], cwd=WEB_DIR, check=True)
        logger.info("Building the web app...")
        subprocess.run([npm, "run", "build"], cwd=WEB_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Web app build failed: {e}")
        return has_build


def start_services():
    """
    Builds the web app if needed and starts the FastAPI backend, which also serves the web app.
    """
    if not build_web():
        return

    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(".venv", "bin", "python")
    if not os.path.exists(venv_python):
        logger.warning("Virtual environment not found. Running with system python.")
        venv_python = sys.executable

    logger.info("Starting VBCUA...")
    backend_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    time.sleep(2)
    if backend_proc.poll() is not None:
        logger.error("The backend failed to start. Please check the terminal logs above.")
        return

    logger.info("VBCUA is running at http://127.0.0.1:8000  (API docs: http://127.0.0.1:8000/docs)")
    logger.info("Press Ctrl+C to stop.")
    try:
        while backend_proc.poll() is None:
            time.sleep(1)
        logger.error("The backend process stopped unexpectedly.")
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
        logger.info("Stopped.")


if __name__ == "__main__":
    check_env()
    setup_db()
    start_services()
