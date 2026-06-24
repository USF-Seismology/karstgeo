#!/usr/bin/env python3
"""
62_wrapper_compute_real_synthetic_trace_metrics.py

Wrapper for:
    62_compute_real_synthetic_trace_metrics.py

Computes per-trace, per-window, and per-shot metrics for real Geode SEG-2
files 3005.dat..3046.dat versus synthetic no-void SPECFEM2D SU files.

Run:

    python 62_wrapper_compute_real_synthetic_trace_metrics.py
"""

from pathlib import Path
import subprocess
import sys


BASE = Path(
    "/Users/thompsong/Library/CloudStorage/Box-Box/thompsong/"
    "2026KarstGeophysicsDEP"
)

SOURCES_GROUNDED = (
    BASE
    / "02_Modelling/Seismic/specfem2d/felix/9_LayerModel_Simulation/SOURCES_GROUNDED"
)

NO_VOID_MODEL = SOURCES_GROUNDED / "9_LAYER_MODEL_TOPO_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"

DATA_DIR = NO_VOID_MODEL / "DATA"

REAL_DIR = (
    BASE
    / "04_FieldData/051826/051826_Seismics_T1"
)

OUTPUT_DIR = (
    BASE
    / "02_Modelling/Seismic/differencing/real_vs_synthetic_novoid_metrics"
)

# Use vertical synthetic by default.
# Change to "Ux_file_single_v.su" for horizontal.
COMPONENT_FILE = "Uz_file_single_v.su"

SCRIPT = Path(__file__).with_name("62_compute_real_synthetic_trace_metrics_v2.py")

cmd = [
    sys.executable,
    str(SCRIPT),

    "--data-dir", str(DATA_DIR),
    "--synthetic-novoid-dir", str(NO_VOID_MODEL),
    "--real-dir", str(REAL_DIR),
    "--output-dir", str(OUTPUT_DIR),

    "--component-file", COMPONENT_FILE,

    "--real-first-file", "3005",
    "--real-last-file", "3046",

    "--real-shot-first-x-m", "82.5",
    "--real-shot-dx-m", "2",
    "--real-shot-duplicate-x-m", "102.5",
    "--real-shot-duplicate-files", "3015,3016",
    "--shot-match-tolerance-m", "0.05",

    "--receiver-x-min", "87",
    "--receiver-x-max", "158",
    "--real-first-trace-x-m", "87",
    "--real-dx-m", "1",

    "--analysis-tmin", "0.0",
    "--analysis-tmax", "0.6",

    # Windows used in CSV metrics.
    # pre is an approximate within-gather noise/pre-arrival window;
    # early is a first-arrival/direct-arrival candidate scaling window.
    "--windows", "full:0:0.6,pre:0:0.02,early:0.02:0.12,mid:0.12:0.30,late:0.30:0.60",
    "--preferred-scale-window", "early",
    "--noise-window-name", "pre",

    "--max-freq-hz", "150",

    # Current default processing:
    # demean + linear detrend only. No taper/filter unless uncommented below.
    "--demean",
    "--detrend",
    "--taper-fraction", "0.05",

    # Optional filters to test later:
    # "--bandpass", "5,150",
    "--highpass-hz", "10",

    # Try a quick test first by uncommenting:
    # "--limit", "3",
]

print("\nRunning:\n")
print(" ".join(cmd))
print()

subprocess.run(cmd, check=True)
