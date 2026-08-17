import subprocess
import sys
import os
import time
import argparse
import platform
import shutil

# Add src to path to import platform_resolver
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, "notebook", "src"))
import platform_resolver

def check_command(cmd):
    """Check if a command exists on the system."""
    return shutil.which(cmd) is not None

def check_docker_running():
    """Verify docker daemon is responsive."""
    try:
        # Run docker info and discard output, if it returns 0, daemon is up
        result = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def platform_diagnostics():
    """Outputs the diagnostic information for the current platform."""
    print("=========================================")
    print("PLATFORM DIAGNOSTICS")
    print("=========================================")
    info = platform_resolver.get_platform_info()
    print(f"Platform: {info['platform']}")
    print(f"Architecture: {info['machine']}")
    print(f"Python Version: {info['python_version']}")
    print(f"Python Path: {platform_resolver.get_python_exe(root_dir)}")
    
    is_jetson = info['platform'].lower() == "linux" and info['machine'].lower() in ["aarch64", "arm64"]
    if is_jetson:
        print("\n-> JETSON ORIN / ARM64 LINUX DETECTED")
        print("-> Using Jetson installation mode.")
    elif info['platform'].lower() == "linux":
        print("\n-> STANDARD X86_64 LINUX DETECTED")
    elif info['platform'].lower() == "windows":
        print("\n-> WINDOWS DETECTED")
        
    print("=========================================\n")

def run_installation():
    """Installs required dependencies safely based on platform."""
    print("=========================================")
    print("RUNNING EDGEVISION INSTALLER")
    print("=========================================")
    
    python_exe = platform_resolver.get_python_exe(root_dir)
    info = platform_resolver.get_platform_info()
    is_jetson = info['platform'].lower() == "linux" and info['machine'].lower() in ["aarch64", "arm64"]
    is_linux = info['platform'].lower() == "linux"
    
    # 1. System Dependencies (Linux only)
    if is_linux:
        missing = []
        if not check_command("docker"):
            missing.append("docker.io")
        if not check_command("docker-compose") and not check_command("docker"):
            missing.append("docker-compose")
        if not check_command("npm"):
            missing.extend(["nodejs", "npm"])
            
        if missing:
            print("\n[!] WARNING: Missing required system dependencies for EdgeVision:")
            for m in missing:
                print(f"    - {m}")
            choice = input("\nWould you like to automatically install them via apt-get? (Y/N): ").strip().lower()
            if choice == 'y':
                try:
                    subprocess.run(["sudo", "apt-get", "update"], check=True)
                    subprocess.run(["sudo", "apt-get", "install", "-y"] + missing, check=True)
                    print("-> System dependencies installed successfully!\n")
                except subprocess.CalledProcessError:
                    print("\n[X] Error: Failed to install dependencies via apt-get.")
                    sys.exit(1)
            else:
                print("\n[X] Cannot proceed without required system dependencies. Exiting.")
                sys.exit(1)
    elif info['platform'].lower() == "windows":
        if not check_command("docker"):
            print("\n[X] CRITICAL ERROR: Docker is not installed or not in PATH!")
            print("Please install Docker Desktop for Windows and try again.")
            sys.exit(1)
        if not check_command("npm"):
            print("\n[X] CRITICAL ERROR: Node.js/NPM is not installed!")
            print("Please install Node.js and try again.")
            sys.exit(1)

    # 2. Python Dependencies
    print("\n-> Installing Common Python dependencies...")
    req_common = os.path.join(root_dir, "notebook", "requirements-common.txt")
    if os.path.exists(req_common):
        subprocess.run([python_exe, "-m", "pip", "install", "-r", req_common], check=True)
    else:
        print("-> Warning: notebook/requirements-common.txt not found!")

    if is_jetson:
        print("\n-> [JETSON MODE] Skipping PyPI PyTorch installation.")
        print("-> IMPORTANT: Jetson devices must use NVIDIA's custom PyTorch/Torchvision wheels!")
        print("-> Ensure JetPack 5.x+ is flashed. If PyTorch is missing, install it manually via NVIDIA docs.")
    elif is_linux:
        print("\n-> Installing Standard Linux PyTorch dependencies...")
        req_linux = os.path.join(root_dir, "notebook", "requirements-linux.txt")
        if os.path.exists(req_linux):
            subprocess.run([python_exe, "-m", "pip", "install", "-r", req_linux], check=True)
    else:
        print("\n-> Installing Windows PyTorch dependencies...")
        req_windows = os.path.join(root_dir, "notebook", "requirements-windows.txt")
        if os.path.exists(req_windows):
            subprocess.run([python_exe, "-m", "pip", "install", "-r", req_windows], check=True)

    # 3. NPM Dependencies
    print("\n-> Installing Next.js dependencies...")
    frontend_dir = os.path.join(root_dir, "notebook", "frontend")
    subprocess.run(["npm", "install"], cwd=frontend_dir, shell=platform_resolver.is_windows(), check=True)
    print("\n-> INSTALLATION COMPLETE.")

def verify_system():
    """Generates a hardware/environment verification report."""
    print("=========================================")
    print("VERIFYING SYSTEM CONFIGURATION")
    print("=========================================")
    python_exe = platform_resolver.get_python_exe(root_dir)
    
    # Run a tiny python script inside the environment to check torch
    check_script = """
import platform
import sys

print(f"Python: {sys.version.split(' ')[0]}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")
except Exception as e:
    print(f"PyTorch: ERROR - {e}")
    """
    
    result = subprocess.run([python_exe, "-c", check_script], capture_output=True, text=True)
    print(result.stdout)
    
    # Save the report for Jetson
    info = platform_resolver.get_platform_info()
    if info['platform'].lower() == "linux" and info['machine'].lower() in ["aarch64", "arm64"]:
        report_dir = os.path.join(root_dir, "notebook", "outputs", "deployment")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "JETSON_ENVIRONMENT_REPORT.md")
        with open(report_path, "w") as f:
            f.write("# Jetson Hardware Validation Report\n\n")
            f.write(result.stdout)
        print(f"-> Jetson report saved to: {report_path}")

def start_servers(video_arg=None):
    """Starts the Database, Backend, and Frontend."""
    print("=========================================")
    print("Starting EdgeVision Servers...")
    python_exe = platform_resolver.get_python_exe(root_dir)
    print(f"Using environment: {python_exe}")
    print("=========================================\n")

    backend_dir = os.path.join(root_dir, "notebook", "backend")
    frontend_dir = os.path.join(root_dir, "notebook", "frontend")
    scripts_dir = os.path.join(backend_dir, "scripts")
    src_dir = os.path.join(root_dir, "notebook", "src")

    print("-> Checking Docker Daemon...")
    if not check_docker_running():
        print("\n[X] CRITICAL ERROR: Docker daemon is NOT running!")
        sys.exit(1)

    print("-> Starting PostgreSQL Database (Docker)...")
    compose_cmd = ["docker-compose"] if check_command("docker-compose") else ["docker", "compose"]
    try:
        subprocess.run(
            compose_cmd + ["up", "-d", "db"],
            cwd=os.path.join(root_dir, "notebook"),
            shell=platform_resolver.is_windows(),
            check=True
        )
    except subprocess.CalledProcessError:
        print("\n[X] Error: Failed to start database container.")
        sys.exit(1)
        
    time.sleep(3)

    print("-> Initializing Database...")
    subprocess.run([python_exe, "init_db.py"], cwd=scripts_dir)

    print("-> Starting FastAPI Backend...")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--port", "8000", "--reload"],
        cwd=backend_dir
    )

    time.sleep(2)

    print("-> Starting Next.js Frontend...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=platform_resolver.is_windows()
    )

    pipeline_process = None
    if video_arg:
        print(f"-> Starting ML Pipeline with video: {video_arg}...")
        pipeline_process = subprocess.Popen(
            [python_exe, "pipeline.py", "--video", os.path.abspath(video_arg)],
            cwd=src_dir
        )

    try:
        print("\n=========================================")
        print("Servers are running concurrently!")
        print("-> Frontend Dashboard: http://localhost:3000")
        print("-> Backend API Docs:   http://localhost:8000/docs")
        print("Press Ctrl+C in this terminal to stop everything.")
        print("=========================================\n")
        
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


def main():
    parser = argparse.ArgumentParser(description="EdgeVision Deployment Manager")
    parser.add_argument("--platform-check", action="store_true", help="Perform platform diagnostics only")
    parser.add_argument("--install", action="store_true", help="Install platform-safe dependencies")
    parser.add_argument("--verify", action="store_true", help="Verify hardware and database environment")
    parser.add_argument("--run", action="store_true", help="Start the EdgeVision application")
    parser.add_argument("--video", type=str, help="Path to input video to run the ML pipeline automatically")
    
    args = parser.parse_args()

    # If no flags provided, ask for legacy choice
    if not (args.platform_check or args.install or args.verify or args.run):
        print("=========================================")
        print("EdgeVision Deployment Manager (Legacy Mode)")
        print("=========================================")
        print("1. Install Dependencies & Run")
        print("2. Run Only")
        choice = ""
        while choice not in ["1", "2"]:
            choice = input("Enter choice (1 or 2): ").strip()
            
        if choice == "1":
            platform_diagnostics()
            run_installation()
            start_servers(args.video)
        else:
            platform_diagnostics()
            start_servers(args.video)
        return

    # Execute requested flags
    if args.platform_check:
        platform_diagnostics()
    if args.install:
        run_installation()
    if args.verify:
        verify_system()
    if args.run:
        start_servers(args.video)

if __name__ == "__main__":
    main()
