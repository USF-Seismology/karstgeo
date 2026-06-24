#!/usr/bin/env python3
"""
60_wrapper_v5.py

Wrapper for:
    60_compare_synthetic_cave_no_cave_v4.py

Compares paired synthetic SPECFEM2D shot gathers:

    WITH CAVE / VOID
    WITHOUT CAVE / NO VOID

This version uses the corrected current model path:

    .../felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED/

Run:

    python 60_wrapper_v5.py
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
    / "02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED"
)

CAVE_MODEL = SOURCES_GROUNDED / "9_LAYER_MODEL_TOPO_VOID_150m_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"
NO_VOID_MODEL = SOURCES_GROUNDED / "9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"

DATA_DIR = NO_VOID_MODEL / "DATA"

# Choose "Uz_file_single_v.su" or "Ux_file_single_v.su"
COMPONENT_FILE = "Ux_file_single_v.su"

COMPONENT_NAME = "Ux" if COMPONENT_FILE.startswith("Ux") else "Uz"

OUTPUT_DIR = (
    BASE
    / "02_Modelling/Seismic/differencing/synthetic_cave_vs_nocave_comparison"
    / COMPONENT_NAME
)

SCRIPT = Path(__file__).with_name("60_compare_synthetic_cave_no_cave_v4.py")

cmd = [
    sys.executable,
    str(SCRIPT),

    "--data-dir", str(DATA_DIR),
    "--cave-dir", str(CAVE_MODEL),
    "--nocave-dir", str(NO_VOID_MODEL),
    "--output-dir", str(OUTPUT_DIR),

    "--cave-pattern", f"SURVEY_OUTPUT/**/{COMPONENT_FILE}",
    "--nocave-pattern", f"SURVEY_OUTPUT/**/{COMPONENT_FILE}",

    "--pair-mode", "order",

    "--max-freq-hz", "150",

    "--write-diff-segy",
    "--write-individual-wiggles",

    "--write-overlay-wiggles",
    "--overlay-normalize", "pair",
    "--overlay-wiggle-scale", "0.45",
    "--peak-scale-halfwidth-s", "0.015",

    "--cave-extent-x-m", "122,130",

    # Uncomment for a quick test:
    # "--limit", "3",
]

print("\nChecking key paths:")
print(f"  DATA_DIR:          {DATA_DIR}")
print(f"  STATIONS:          {DATA_DIR / 'STATIONS'}")
print(f"  SOURCES_LIST:      {DATA_DIR / 'SOURCES_LIST.txt'}")
print(f"  CAVE_MODEL:        {CAVE_MODEL}")
print(f"  NO_VOID_MODEL:     {NO_VOID_MODEL}")
print(f"  CAVE OUTPUT:       {CAVE_MODEL / 'SURVEY_OUTPUT'}")
print(f"  NO VOID OUTPUT:    {NO_VOID_MODEL / 'SURVEY_OUTPUT'}")
print(f"  SCRIPT:            {SCRIPT}")

missing = [
    p for p in [
        DATA_DIR / "STATIONS",
        DATA_DIR / "SOURCES_LIST.txt",
        CAVE_MODEL,
        NO_VOID_MODEL,
        CAVE_MODEL / "SURVEY_OUTPUT",
        NO_VOID_MODEL / "SURVEY_OUTPUT",
        SCRIPT,
    ]
    if not p.exists()
]

if missing:
    print("\nWARNING: These paths do not exist:")
    for p in missing:
        print(f"  {p}")
    print("\nContinuing anyway; the main script will try to resolve DATA automatically.")

print("\nRunning synthetic cave/no-cave comparison:\n")
print("\n".join(cmd))
print()

subprocess.run(cmd, check=True)
