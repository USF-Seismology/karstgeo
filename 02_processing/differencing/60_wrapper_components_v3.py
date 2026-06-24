#!/usr/bin/env python3
"""
60_wrapper_components.py

Run synthetic cave/no-cave comparison for both SPECFEM components:

    Ux_file_single_v.su
    Uz_file_single_v.su

Each component writes to its own output subfolder:

    synthetic_cave_vs_nocave_comparison/Ux
    synthetic_cave_vs_nocave_comparison/Uz

Run:

    python 60_wrapper_components.py
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
PAR_FILE = CAVE_MODEL / "DATA" / "Par_file"

OUTPUT_ROOT = (
    BASE
    / "02_Modelling/Seismic/differencing/synthetic_cave_vs_nocave_comparison"
)

SCRIPT = Path(__file__).with_name("60_compare_synthetic_cave_no_cave_v7.py")

COMPONENT_FILES = [
    "Ux_file_single_v.su",
    "Uz_file_single_v.su",
]


def component_name(component_file: str) -> str:
    if component_file.startswith("Ux"):
        return "Ux"
    if component_file.startswith("Uz"):
        return "Uz"
    return Path(component_file).stem


def check_common_paths() -> None:
    print("\nChecking common paths:")
    print(f"  DATA_DIR:          {DATA_DIR}")
    print(f"  STATIONS:          {DATA_DIR / 'STATIONS'}")
    print(f"  SOURCES_LIST:      {DATA_DIR / 'SOURCES_LIST.txt'}")
    print(f"  CAVE_MODEL:        {CAVE_MODEL}")
    print(f"  NO_VOID_MODEL:     {NO_VOID_MODEL}")
    print(f"  CAVE OUTPUT:       {CAVE_MODEL / 'SURVEY_OUTPUT'}")
    print(f"  NO VOID OUTPUT:    {NO_VOID_MODEL / 'SURVEY_OUTPUT'}")
    print(f"  PAR_FILE:          {PAR_FILE}")
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
            PAR_FILE,
        ]
        if not p.exists()
    ]

    if missing:
        print("\nWARNING: These common paths do not exist:")
        for p in missing:
            print(f"  {p}")
        print("\nContinuing anyway; the main script will try to resolve DATA automatically.")


def run_component(component_file: str) -> None:
    comp = component_name(component_file)
    output_dir = OUTPUT_ROOT / comp

    cmd = [
        sys.executable,
        str(SCRIPT),

        "--data-dir", str(DATA_DIR),
        "--par-file", str(PAR_FILE),
        "--cave-dir", str(CAVE_MODEL),
        "--nocave-dir", str(NO_VOID_MODEL),
        "--output-dir", str(output_dir),

        "--cave-pattern", f"SURVEY_OUTPUT/**/{component_file}",
        "--nocave-pattern", f"SURVEY_OUTPUT/**/{component_file}",

        "--pair-mode", "order",

        "--max-freq-hz", "150",

        "--write-diff-segy",
        "--write-individual-wiggles",

        "--write-overlay-wiggles",
        "--overlay-normalize", "pair",
        "--overlay-wiggle-scale", "0.45",
        "--peak-scale-halfwidth-s", "0.015",

        # Trace-normalized secondary figures are on by default in v5.
        "--write-trace-normalized-figures",
        "--trace-normalize-method", "rms",

        "--write-band-energy",
        "--band-energy-bands", "10-30,30-80,80-150",
        "--band-energy-window-s", "0.05",
        "--band-energy-step-s", "0.01",
        "--band-energy-normalize-per-trace",

        "--cave-extent-x-m", "122,130",

        # Uncomment for a quick test:
        # "--limit", "3",
    ]

    print(f"\n{'=' * 80}")
    print(f"Running synthetic cave/no-cave comparison for {comp}")
    print(f"{'=' * 80}")

    print("\nComponent-specific paths:")
    print(f"  component_file:    {component_file}")
    print(f"  output_dir:        {output_dir}")
    print(f"  cave glob:         {CAVE_MODEL / 'SURVEY_OUTPUT'}/**/{component_file}")
    print(f"  no-void glob:      {NO_VOID_MODEL / 'SURVEY_OUTPUT'}/**/{component_file}")

    print("\nCommand:\n")
    print("\n".join(cmd))
    print()

    subprocess.run(cmd, check=True)


def main() -> int:
    check_common_paths()
    for component_file in COMPONENT_FILES:
        run_component(component_file)
    print("\nCompleted all synthetic cave/no-cave components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
