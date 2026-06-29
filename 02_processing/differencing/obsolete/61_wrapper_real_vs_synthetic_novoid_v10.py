#!/usr/bin/env python3
"""
61_wrapper_real_vs_synthetic_novoid_components.py

Run real-vs-synthetic-no-void comparison for both SPECFEM components:

    Ux_file_single_v.su
    Uz_file_single_v.su

Each component writes to its own output subfolder:

    real_vs_synthetic_novoid_comparison/Ux
    real_vs_synthetic_novoid_comparison/Uz

Run:

    python 61_wrapper_real_vs_synthetic_novoid_components.py
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

NO_VOID_MODEL = SOURCES_GROUNDED / "9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"

DATA_DIR = NO_VOID_MODEL / "DATA"

REAL_DIR = (
    BASE
    / "04_FieldData/051826/051826_Seismics_T1"
)

OUTPUT_ROOT = (
    BASE
    / "02_Modelling/Seismic/differencing/real_vs_synthetic_novoid_comparison"
)

SCRIPT = Path(__file__).with_name("61_compare_real_geode_vs_synthetic_novoid_v10.py")

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
    print(f"  DATA_DIR:            {DATA_DIR}")
    print(f"  STATIONS:            {DATA_DIR / 'STATIONS'}")
    print(f"  SOURCES_LIST:        {DATA_DIR / 'SOURCES_LIST.txt'}")
    print(f"  NO_VOID_MODEL:       {NO_VOID_MODEL}")
    print(f"  SYNTHETIC OUTPUT:    {NO_VOID_MODEL / 'SURVEY_OUTPUT'}")
    print(f"  REAL_DIR:            {REAL_DIR}")
    print(f"  SCRIPT:              {SCRIPT}")

    missing = [
        p for p in [
            DATA_DIR / "STATIONS",
            DATA_DIR / "SOURCES_LIST.txt",
            NO_VOID_MODEL,
            NO_VOID_MODEL / "SURVEY_OUTPUT",
            REAL_DIR,
            SCRIPT,
        ]
        if not p.exists()
    ]

    if missing:
        print("\nWARNING: These common paths do not exist:")
        for p in missing:
            print(f"  {p}")
        print("\nContinuing anyway; the main script may still resolve alternate geometry paths.")


def run_component(component_file: str) -> None:
    comp = component_name(component_file)
    output_dir = OUTPUT_ROOT / comp

    cmd = [
        sys.executable,
        str(SCRIPT),

        "--data-dir", str(DATA_DIR),
        "--synthetic-novoid-dir", str(NO_VOID_MODEL),
        "--real-dir", str(REAL_DIR),
        "--output-dir", str(output_dir),

        "--real-first-file", "3005",
        "--real-last-file", "3046",

        "--real-shot-first-x-m", "82.5",
        "--real-shot-dx-m", "2",
        "--real-shot-duplicate-x-m", "102.5",
        "--real-shot-duplicate-files", "3015,3016",
        "--shot-match-tolerance-m", "0.05",

        "--component-file", component_file,

        "--receiver-x-min", "87",
        "--receiver-x-max", "158",
        "--real-first-trace-x-m", "87",
        "--real-dx-m", "1",

        "--tmin", "0.0",
        "--tmax", "0.4",
        "--max-freq-hz", "150",

        "--synthetic-time-shift-ms", "-31.6",

        "--scale-mode", "fixed",
        "--fixed-scale-factor", "2.96e7",
        "--scale-tmin", "0.02",
        "--scale-tmax", "0.12",

        "--demean",
        "--detrend",
        "--taper-fraction", "0.05",
        "--highpass-hz", "10",
        "--filter-corners", "4",
        "--zerophase",

        "--cave-extent-x-m", "122,130",

        "--write-individual-wiggles",
        "--write-overlay-wiggles",
        "--overlay-normalize", "pair",
        "--overlay-wiggle-scale", "0.45",
        "--peak-scale-halfwidth-s", "0.015",

        # Trace-normalized secondary figures are on by default in v10.
        "--write-trace-normalized-figures",
        "--trace-normalize-method", "rms",

        # Uncomment if desired:
        # "--write-diff-segy",

        # Uncomment for quick testing:
        # "--limit", "3",
    ]

    print(f"\n{'=' * 80}")
    print(f"Running real vs synthetic no-void comparison for {comp}")
    print(f"{'=' * 80}")

    print("\nComponent-specific paths:")
    print(f"  component_file:    {component_file}")
    print(f"  output_dir:        {output_dir}")
    print(f"  synthetic glob:    {NO_VOID_MODEL / 'SURVEY_OUTPUT'}/**/{component_file}")

    print("\nCommand:\n")
    print("\n".join(cmd))
    print()

    subprocess.run(cmd, check=True)


def main() -> int:
    check_common_paths()
    for component_file in COMPONENT_FILES:
        run_component(component_file)
    print("\nCompleted all real-vs-synthetic components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
