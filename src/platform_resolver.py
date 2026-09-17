import os
import sys
import platform

def is_linux():
    return platform.system().lower() == "linux"

def is_windows():
    return platform.system().lower() == "windows"

def get_platform_info():
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version()
    }

def get_python_exe(root_dir=None):
    """
    Returns the absolute path to the virtual environment Python executable,
    handling cross-platform differences between Windows and Linux.
    """
    if root_dir is None:
        # If not provided, assume root_dir is two levels up from this file (Tfrenzy root)
        # Assuming this file is in notebook/src/platform_resolver.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    venv_python_win = os.path.join(root_dir, "ppe-env", "Scripts", "python.exe")
    venv_python_lin = os.path.join(root_dir, "ppe-env", "bin", "python")
    
    if os.path.exists(venv_python_win):
        return venv_python_win
    elif os.path.exists(venv_python_lin):
        return venv_python_lin
    else:
        # Fallback to current global executable if venv not found
        return sys.executable
