#!/usr/bin/env python3
"""
61_wrapper_real_vs_synthetic_novoid.py

Run real Geode SEG-2 vs synthetic no-void comparison.

This assumes:
    real files: 3005.dat..3046.dat
    real shot x: 3005=82.5 m, +2 m nominal spacing
    correction: 3015.dat and 3016.dat were both x=102.5 m
    real receiver x: 87..158 m
    synthetic: NO_VOID_MODEL/SURVEY_OUTPUT/**/Uz_file_single_v.su
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

REAL_DIR = BASE / "04_FieldData/051826/051826_Seismics_T1"

OUTPUT_DIR = BASE / "02_Modelling/Seismic/differencing/real_vs_synthetic_novoid_comparison"

SCRIPT = Path(__file__).with_name("61_compare_real_geode_vs_synthetic_novoid_v4.py")

# Use Uz for vertical synthetic output. Change to Ux_file_single_v.su for horizontal.
COMPONENT_FILE = "Uz_file_single_v.su"

cmd = [
    sys.executable,
    str(SCRIPT),
    "--data-dir", str(DATA_DIR),
    "--synthetic-novoid-dir", str(NO_VOID_MODEL),
    "--real-dir", str(REAL_DIR),
    "--output-dir", str(OUTPUT_DIR),

    "--real-first-file", "3005",
    "--real-last-file", "3046",
    "--real-shot-first-x-m", "82.5",
    "--real-shot-dx-m", "2",
    "--real-shot-duplicate-x-m", "102.5",
    "--real-shot-duplicate-files", "3015,3016",
    "--shot-match-tolerance-m", "0.05",

    "--component-file", COMPONENT_FILE,

    "--receiver-x-min", "87",
    "--receiver-x-max", "158",
    "--real-first-trace-x-m", "87",
    "--real-dx-m", "1",

    "--tmin", "0.0",
    "--tmax", "0.4",
    "--max-freq-hz", "150",

    # Adopt current best global timing and amplitude corrections
    # from lag-aligned direct-arrival-window diagnostics.
    "--synthetic-time-shift-ms", "-31.6",
    "--scale-mode", "fixed",
    "--fixed-scale-factor", "2.96e7",
    "--scale-tmin", "0.02",
    "--scale-tmax", "0.12",

    # Same gentle processing used by metrics/lag scripts.
    "--demean",
    "--detrend",
    "--taper-fraction", "0.05",
    "--highpass-hz", "10",
    "--filter-corners", "4",
    "--zerophase",

    "--cave-extent-x-m", "122,130",

    "--write-individual-wiggles",
    # "--write-diff-segy",
]

print("\nRunning:\n")
print(" ".join(cmd))
print()
subprocess.run(cmd, check=True)
