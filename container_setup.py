import os
import subprocess
import sys


TENSORFLOW_REQUIREMENTS_FILENAME = "tensorflow_requirements.txt"


def initialize_tf_requirements():
    """Initialize tensorflow container to install extra dependencies."""
    tensorflow_requirements_full_path = os.path.join(os.path.dirname(__file__), TENSORFLOW_REQUIREMENTS_FILENAME)
    if os.path.exists(tensorflow_requirements_full_path):
        print(f"Installing dependencies from {tensorflow_requirements_full_path}, {sys.executable}")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", tensorflow_requirements_full_path],
            capture_output=True,
            encoding="utf-8",
        )
        print(process.stderr)
        print(process.stdout)
    else:
        print(f"Skipping dependency installation. {tensorflow_requirements_full_path} file not found.")
