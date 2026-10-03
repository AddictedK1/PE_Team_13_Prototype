"""Root entrypoint for Streamlit application."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Delegate directly to UI implementation
from app.ui import *  # noqa: F401, F403
