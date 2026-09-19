"""Path bootstrap for agents/ scripts after repo reorganization."""
import os
import sys

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(AGENTS_DIR)

for _p in (
    AGENTS_DIR,
    REPO_ROOT,
    os.path.join(REPO_ROOT, "python"),
    os.path.join(REPO_ROOT, "training"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Prefer running with repo root as CWD so relative runtime/ paths stay stable.
os.chdir(REPO_ROOT)
