import subprocess
import sys
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="EdgeVision App Launcher")
    parser.add_argument("--video", type=str, help="Path to input video to run the ML pipeline automatically")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "notebook", "backend")
    frontend_dir = os.path.join(root_dir, "notebook", "frontend")
    scripts_dir = os.path.join(backend_dir, "scripts")
    src_dir = os.path.join(root_dir, "notebook", "src")

    # Look for virtual environment python (Windows vs Linux)
    venv_python_win = os.path.join(root_dir, "ppe-env", "Scripts", "python.exe")
    venv_python_lin = os.path.join(root_dir, "ppe-env", "bin", "python")
    
    if os.path.exists(venv_python_win):
        python_exe = venv_python_win
    elif os.path.exists(venv_python_lin):
        python_exe = venv_python_lin
    else:
        python_exe = sys.executable

    print("=========================================")
    print("EdgeVision Deployment Manager")
    print("=========================================")
    print("Select deployment mode:")
    print("1. New Deployment (Install dependencies & run)")
    print("2. Old Deployment (Just run)")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\n=========================================")
        print("Running Installations...")
        print("=========================================")
        
        # Install Python dependencies
        req_path = os.path.join(root_dir, "notebook", "requirements.txt")
        if os.path.exists(req_path):
            print("-> Installing Python dependencies...")
            subprocess.run([python_exe, "-m", "pip", "install", "-r", req_path], check=True)
        else:
            print("-> Warning: notebook/requirements.txt not found!")
            
        # Install NPM dependencies
        print("-> Installing Next.js dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True, check=True)
        print("Installations completed.\n")

    print("=========================================")
    print("Starting EdgeVision Servers...")
    if python_exe != sys.executable:
        print(f"Using virtual environment: {python_exe}")
    print("=========================================\n")

    # 0. Start Database
    print("-> Starting PostgreSQL Database (Docker)...")
    subprocess.run(
        ["docker-compose", "up", "-d", "db"],
        cwd=os.path.join(root_dir, "notebook"),
        shell=True
    )
    time.sleep(3) # Give DB a moment to initialize

    # 1. Init Database
    print("-> Initializing Database...")
    subprocess.run(
        [python_exe, "init_db.py"],
        cwd=scripts_dir
    )

    # 2. Start Backend
    print("-> Starting FastAPI Backend...")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--port", "8000", "--reload"],
        cwd=backend_dir
    )

    # Brief pause to let backend initialize
    time.sleep(2)

    # 3. Start Frontend
    print("-> Starting Next.js Frontend...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )

    pipeline_process = None
    if args.video:
        print(f"-> Starting ML Pipeline with video: {args.video}...")
        pipeline_process = subprocess.Popen(
            [python_exe, "pipeline.py", "--video", os.path.abspath(args.video)],
            cwd=src_dir
        )

    try:
        print("\n=========================================")
        print("Servers are running concurrently!")
        print("-> Frontend Dashboard: http://localhost:3000")
        print("-> Backend API Docs:   http://localhost:8000/docs")
        if args.video:
            print("-> ML Pipeline is running in the background.")
        else:
            print("Tip: To run the pipeline, restart this script with --video <path>")
        print("=========================================")
        print("Press Ctrl+C in this terminal to stop everything.\n")
        
        # Wait indefinitely until user terminates
        backend_process.wait()
        frontend_process.wait()
        if pipeline_process:
            pipeline_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down servers gracefully...")
        backend_process.terminate()
        frontend_process.terminate()
        if pipeline_process:
            pipeline_process.terminate()
            
        backend_process.wait()
        frontend_process.wait()
        if pipeline_process:
            pipeline_process.wait()
            
        print("Servers stopped.")

if __name__ == "__main__":
    main()
