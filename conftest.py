"""Ensures the repo root is importable as a package root (for `api.*`,
`src.*`) regardless of which directory pytest is invoked from."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
