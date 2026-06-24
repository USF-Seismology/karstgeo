#!/usr/bin/env python3
"""
60_wrapper.py

Wrapper for 60_compare_synthetic_cave_no_cave.py.

This version is for the SPECFEM2D folder layout where each single-shot output
is below:

    MODEL_ROOT/SURVEY_OUTPUT/<single-shot-folder>/Uz_file_single_v.su
    MODEL_ROOT/SURVEY_OUTPUT/<single-shot-folder>/Ux_file_single_v.su

By default this compares Uz. Change COMPONENT_FILE to "Ux_file_single_v.su"
to compare horizontal displacement.
"""

from pathlib import Path
import subprocess
import sys


# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------

BOX_SEISMIC = Path(
    "~/Library/CloudStorage/Box-Box/thompsong/"
    "2026KarstGeophysicsDEP/02_Modelling/Seismic"
).expanduser()

MODEL_BASE = BOX_SEISMIC / (
    "specfem2d/felix/9_LayerModel_Simulation/SOURCES_GROUNDED"
)

CAVE_ROOT = MODEL_BASE / (
    "9_LAYER_MODEL_TOPO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"
)

NO_CAVE_ROOT = MODEL_BASE / (
    "9_LAYER_MODEL_TOPO_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"
)

# Use the cave model DATA folder as authoritative geometry.
# STATIONS should be identical between cave/no-cave runs.
DATA_DIR = CAVE_ROOT / "DATA"

# Compare vertical displacement by default.
# Change to "Ux_file_single_v.su" for horizontal displacement.
#COMPONENT_FILE = "Uz_file_single_v.su"
COMPONENT_FILE = "Ux_file_single_v.su"

OUTPUT_DIR = (
    BOX_SEISMIC
    / "differencing"
    / "synthetic_cave_vs_nocave_comparison"
    / COMPONENT_FILE.replace("_file_single_v.su", "")
)

# Put this wrapper in the same directory as the main comparison script,
# or change SCRIPT explicitly.
SCRIPT = Path(__file__).with_name("60_compare_synthetic_cave_no_cave_v3.py")


# ------------------------------------------------------------------
# COMMAND
# ------------------------------------------------------------------

cmd = [
    sys.executable,
    str(SCRIPT),

    "--data-dir", str(DATA_DIR),

    "--cave-dir", str(CAVE_ROOT),
    "--nocave-dir", str(NO_CAVE_ROOT),

    "--output-dir", str(OUTPUT_DIR),

    # Recursively find all single-shot SU files under SURVEY_OUTPUT.
    "--cave-pattern", f"SURVEY_OUTPUT/**/{COMPONENT_FILE}",
    "--nocave-pattern", f"SURVEY_OUTPUT/**/{COMPONENT_FILE}",

    # SPECFEM single-shot folders are usually in the same order for both runs.
    # Source x positions are read from DATA/SOURCES_LIST.txt in that same order.
    "--pair-mode", "order",

    "--max-freq-hz", "150",
    "--write-diff-segy",
    "--write-individual-wiggles",
    "--write-overlay-wiggles",
    "--overlay-normalize", "pair",
    "--overlay-wiggle-scale", "0.45",
    "--peak-scale-halfwidth-s", "0.015",

    # Jochen/Pati cave center estimates were around 122-130 m.
    "--cave-extent-x-m", "122,130",
]

print("\nRunning synthetic cave/no-cave comparison:\n")
for item in cmd:
    print(item)
print()

subprocess.run(cmd, check=True)
