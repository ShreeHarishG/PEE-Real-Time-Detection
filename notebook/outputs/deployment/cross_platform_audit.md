# EdgeVision Cross-Platform Deployment Audit

## 1. Installer (`run_app.py`) Audit
**Current Implementation:**
- **OS Detection:** Uses `platform.system().lower()` to check for `"linux"` or `"windows"`.
- **Python Resolution:** Checks for `ppe-env/Scripts/python.exe` (Windows) or `ppe-env/bin/python` (Linux).
- **Dependency Installation:** Blindly executes `pip install -r notebook/requirements.txt` on both Windows and Linux if a new deployment is selected.
- **Docker Detection:** Uses `shutil.which("docker")` and `subprocess.run(["docker", "info"])`. On Linux, it uses `apt-get` to install missing Docker and NPM dependencies.

**Identified Issues:**
1. **Generic PyTorch on Jetson:** Installing `requirements.txt` blindly on Jetson will pull the generic PyTorch CPU/x86 wheels. It lacks the hardware/architecture check (aarch64 / TensorRT / CUDA) required to skip or guide the NVIDIA JetPack installation.
2. **Docker vs Native DB:** `run_app.py` forces Docker Compose for PostgreSQL. On some Jetson deployments, Docker might be avoided in favor of bare-metal installs.

## 2. Backend (`backend/app/main.py`) Audit
**Current Implementation:**
- **Subprocess Creation:** `main.py` explicitly hardcodes the virtual environment path: `venv_python = os.path.join(root_dir, "ppe-env", "Scripts", "python.exe")`.
- **Camera Resolution:** It resolves the camera path using `os.path.join(root_dir, source)` if it's not absolute or RTSP.

**Identified Issues:**
1. **Linux Subprocess Crash:** The hardcoded `Scripts/python.exe` path guarantees that `pipeline.py` will fail to launch on Linux, as the path does not exist. It needs a central Python resolver identical to the one in `run_app.py`.
2. **Camera Backend:** `cv2.CAP_DSHOW` was recently removed, which is good for Linux compatibility (as it's a Windows-only enum), but there is no structured fallback for Linux V4L2/GStreamer.

## 3. Core Requirements (`notebook/requirements.txt`)
**Current Implementation:**
- Single `requirements.txt` containing all packages including `torch`, `torchvision`, and `opencv-python-headless`.

**Identified Issues:**
1. Jetson environments should absolutely not install `torch` from PyPI. Requirements must be split into `-common`, `-windows`, and `-linux` / `-jetson` to prevent breaking existing NVIDIA environments.

## 4. Path Handling (`notebook/src/pipeline.py`)
**Current Implementation:**
- Path construction uses `os.path.join` which is safe cross-platform.
- Some string replacements like `.replace("\\", "/")` are used to convert paths to URLs.

**Identified Issues:**
- Ensure all model paths load dynamically instead of relying on exact hardcoded paths that might change between staging and production.

## 5. Next Steps
Based on this audit, we must:
1. Create a `utils/platform_resolver.py` to handle Python path resolution centrally for both `run_app.py` and `main.py`.
2. Implement Jetson hardware checks (CUDA/Tegra/TensorRT) in `run_app.py` to block dangerous pip installs.
3. Split `requirements.txt`.
4. Add the requested CLI arguments (`--platform-check`, `--install`, `--verify`, `--run`) to `run_app.py`.
