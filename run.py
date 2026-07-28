import subprocess
import sys
import time

def run_services():
    """Starts both FastAPI backend and Streamlit frontend concurrently."""
    print("🚀 Starting Enterprise RAG Services...")

    # Start Backend (FastAPI via Uvicorn)
    print("Starting FastAPI Backend on port 8000...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_process = subprocess.Popen(backend_cmd)
    
    # Wait a few seconds to ensure backend is ready before frontend makes any calls
    time.sleep(4)
    
    # Start Frontend (Streamlit)
    print("Starting Streamlit Frontend on port 8501...")
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"]
    frontend_process = subprocess.Popen(frontend_cmd)

    try:
        # Keep main thread alive while subprocesses run
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("✅ Services stopped successfully.")

if __name__ == "__main__":
    run_services()