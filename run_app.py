import os
import sys
import time
import subprocess
import argparse

def is_windows():
    return os.name == 'nt'

def main():
    parser = argparse.ArgumentParser(description="EdgeVision Simple Startup")
    parser.add_argument("--video", type=str, help="Path to input video to run the ML pipeline automatically")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(root_dir, "notebook")
    
    # 1. Resolve Python Environment (Handles both Windows and Linux)
    if is_windows():
        venv_python = os.path.join(root_dir, "ppe-env", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(root_dir, "ppe-env", "bin", "python")
        
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    # 2. Start Database (Handles newer 'docker compose' and older 'docker-compose')
    print("-> Starting PostgreSQL (Docker)...")
    docker_cmd = ["docker", "compose"] if is_windows() else ["sudo", "docker", "compose"]
    docker_fallback = ["docker-compose"] if is_windows() else ["sudo", "docker-compose"]
    try:
        subprocess.run(docker_cmd + ["up", "-d", "db"], cwd=notebook_dir)
    except FileNotFoundError:
        subprocess.run(docker_fallback + ["up", "-d", "db"], cwd=notebook_dir)

    # 3. Initialize DB Tables
    print("-> Waiting for database to boot up (10 seconds)...")
    time.sleep(10)
    print("-> Initializing Database...")
    subprocess.run([python_exe, "init_db.py"], cwd=os.path.join(notebook_dir, "backend", "scripts"))

    # 4. Start FastAPI
    print("-> Starting FastAPI Backend...")
    backend = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--port", "8000", "--reload"],
        cwd=os.path.join(notebook_dir, "backend")
    )
    time.sleep(2)

    # 5. Start Next.js Frontend
    print("-> Starting Next.js Frontend...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(notebook_dir, "frontend"),
        shell=is_windows()
    )

    # 6. (Optional) Start ML Pipeline
    pipeline = None
    if args.video:
        print(f"-> Starting ML Pipeline: {args.video}...")
        video_path = args.video if args.video.isdigit() or args.video.startswith("rtsp") else os.path.abspath(args.video)
        pipeline = subprocess.Popen(
            [python_exe, "pipeline.py", "--video", video_path],
            cwd=os.path.join(notebook_dir, "src")
        )

    # Keep script running until Ctrl+C
    try:
        print("\n✅ All EdgeVision services started! Press Ctrl+C to stop.")
        frontend.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend.terminate()
        frontend.terminate()
        if pipeline:
            pipeline.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
