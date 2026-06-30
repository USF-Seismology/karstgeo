#!/usr/bin/env python3
"""
63_wrapper_estimate_real_synthetic_lags.py

Wrapper for:
    63_estimate_real_synthetic_lags.py

Estimates gather-wide timing lags between real Geode SEG-2 files
3005.dat..3046.dat and synthetic no-void SPECFEM2D SU files.

Run:

    python 63_wrapper_estimate_real_synthetic_lags.py
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
    / "02_Modelling/Seismic/differencing/real_vs_synthetic_novoid_lag_metrics"
)

COMPONENT_FILE = "Uz_file_single_v.su"

SCRIPT = Path(__file__).with_name("63_estimate_real_synthetic_lags.py")

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
    "--analysis-tmax", "0.4",

    # Correlate in the direct-arrival window.
    "--corr-window", "0.02,0.12",

    # Also compute post-alignment scale factors in the direct-arrival window.
    "--scale-window", "0.02,0.12",

    # Search +/- 50 ms for trigger/model timing offset.
    "--max-lag-ms", "50",

    # Weight shot-level lag by real-trace amplitude in the correlation window.
    "--weight-mode", "real_rms",
    "--selected-lag", "weighted_median",

    # Keep all lags initially; tighten these later if needed.
    "--min-corr-for-shot-lag", "-1.0",
    "--min-abs-corr-for-shot-lag", "0.0",
    "--min-good-traces-for-shot-lag", "8",

    # Gentle default processing.
    "--demean",
    "--detrend",
    "--taper-fraction", "0.05",
    "--highpass-hz", "10",
    "--filter-corners", "4",
    "--zerophase",

    # Uncomment for quick testing.
    # "--limit", "3",
]

print("\nRunning:\n")
print(" ".join(cmd))
print()

subprocess.run(cmd, check=True)
