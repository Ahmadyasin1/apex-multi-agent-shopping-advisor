"""
NexusShop — Launch Script
Run: python run.py
Opens the luxury frontend at http://localhost:8000
"""
import subprocess, sys, os, webbrowser, time
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)

    print("=" * 55)
    print("  NexusShop — Intelligent Shopping Advisor v2.0")
    print("  Elite Multi-Agent AI System")
    print("=" * 55)
    print()
    print("  Starting FastAPI server...")
    print("  Frontend: http://localhost:9000")
    print("  API Docs: http://localhost:9000/docs")
    print("  Press Ctrl+C to stop")
    print()

    def open_browser():
        time.sleep(2.5)
        webbrowser.open("http://localhost:9000")

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--reload",
        "--log-level", "warning"
    ])

if __name__ == "__main__":
    main()
