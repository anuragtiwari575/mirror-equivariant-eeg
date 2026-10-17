"""Single source of truth for project paths. Import this from every script so
all work, code, and outputs land in one place.

Edit PROJECT_ROOT / DATA_* here only; nothing else in the codebase hardcodes a
path.
"""
import os

# All project work, code, and outputs live here (created on first run).
PROJECT_ROOT = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work"

# Raw dataset locations: read-only inputs, kept separate from the work folder.
DATA_2A = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work\BCICIV_2a_gdf"
DATA_2B = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work\BCICIV_2b_gdf"
DATA_PHYSIONET = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work\eeg-motor-movementimagery-dataset-1.0.0"
DATA_HGD = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work\v1.0.2"

# Standard subfolders under the work folder.
SRC     = os.path.join(PROJECT_ROOT, "src")       # source modules
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")   # epoched arrays, reports
RESULTS = os.path.join(PROJECT_ROOT, "results")   # metrics, tables
FIGURES = os.path.join(PROJECT_ROOT, "figures")   # plots
CACHE   = os.path.join(PROJECT_ROOT, "cache")     # intermediate artifacts

_SUBDIRS = [SRC, OUTPUTS, RESULTS, FIGURES, CACHE]


# Preprocessing / epoching defaults (single place to tune).
PREPROC = {
    "tmin": 0.0,        # window start relative to cue, seconds
    "tmax": 4.0,        # window end relative to cue, seconds
    "l_freq": 4.0,      # band-pass low edge, Hz (None to skip)
    "h_freq": 40.0,     # band-pass high edge, Hz (None to skip)
    "resample": None,   # target rate, Hz (None keeps native 250 for 2a/2b)
    "normalize": "trial",  # 'trial' (per-trial, leakage-free) | 'channel' | None
}


def ensure_dirs():
    """Create the work-folder tree if it does not exist yet. Returns the root."""
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    for d in _SUBDIRS:
        os.makedirs(d, exist_ok=True)
    return PROJECT_ROOT
