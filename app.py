"""
app.py
======
Entry point for the Research Paper Q&A Chatbot.

Run options:
  1. Streamlit UI (default):
     $ streamlit run app.py

  2. FastAPI backend:
     $ python app.py --api

  3. Both (development):
     $ python app.py --all
"""

import sys
import subprocess
from pathlib import Path


def run_streamlit():
    """Launch the Streamlit chat interface."""
    streamlit_app = Path(__file__).parent / "ui" / "streamlit_app.py"
    subprocess.run(
        ["streamlit", "run", str(streamlit_app), "--server.port", "8501"],
        check=True,
    )


def run_api():
    """Launch the FastAPI backend."""
    subprocess.run(
        ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )


if __name__ == "__main__":
    if "--api" in sys.argv:
        run_api()
    elif "--all" in sys.argv:
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        run_streamlit()
    else:
        run_streamlit()
