#!/usr/bin/env python3
"""
65_wrapper_synthetic_cave_vs_nocave_components.py

Thin wrapper around the unified engine 65_compare_gather_pairs.py.

Runs synthetic WITH CAVE/VOID vs synthetic WITHOUT CAVE/NO-VOID
for Ux and Uz components.
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
    / "02_Modelling/Seismic/differencing/synthetic_cave_vs_nocave_comparison_v2"
)

SCRIPT = Path(__file__).with_name("65_compare_gather_pairs_v2.py")

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
    for label, path in [
        ("DATA_DIR", DATA_DIR),
        ("STATIONS", DATA_DIR / "STATIONS"),
        ("SOURCES_LIST", DATA_DIR / "SOURCES_LIST.txt"),
        ("PAR_FILE", PAR_FILE),
        ("CAVE_MODEL", CAVE_MODEL),
        ("NO_VOID_MODEL", NO_VOID_MODEL),
        ("CAVE OUTPUT", CAVE_MODEL / "SURVEY_OUTPUT"),
        ("NO VOID OUTPUT", NO_VOID_MODEL / "SURVEY_OUTPUT"),
        ("SCRIPT", SCRIPT),
    ]:
        print(f"  {label:15s}: {path}")

    missing = [
        p for p in [
            DATA_DIR / "STATIONS",
            DATA_DIR / "SOURCES_LIST.txt",
            PAR_FILE,
            CAVE_MODEL / "SURVEY_OUTPUT",
            NO_VOID_MODEL / "SURVEY_OUTPUT",
            SCRIPT,
        ]
        if not p.exists()
    ]

    if missing:
        print("\nWARNING: These common paths do not exist:")
        for p in missing:
            print(f"  {p}")


def run_component(component_file: str) -> None:
    comp = component_name(component_file)
    output_dir = OUTPUT_ROOT / comp

    cmd = [
        sys.executable,
        str(SCRIPT),

        "--mode", "synthetic_vs_synthetic",

        "--data-dir", str(DATA_DIR),
        "--par-file", str(PAR_FILE),

        "--reference-dir", str(CAVE_MODEL),
        "--comparison-dir", str(NO_VOID_MODEL),
        "--reference-pattern", f"SURVEY_OUTPUT/**/{component_file}",
        "--comparison-pattern", f"SURVEY_OUTPUT/**/{component_file}",
        "--reference-label", "Synthetic WITH cave/void",
        "--comparison-label", "Synthetic WITHOUT cave/void",

        "--output-dir", str(output_dir),

        "--scale-mode", "none",

        "--max-freq-hz", "150",

        "--write-diff-segy",
        "--write-individual-wiggles",
        "--write-overlay-wiggles",

        "--overlay-normalize", "trace", #"pair",
        "--overlay-wiggle-scale", "0.45",

        "--write-trace-normalized-figures",
        "--trace-normalize-method", "maxabs", #"rms",

        "--write-band-energy",
        "--band-energy-bands", "10-30,30-80,80-150",
        "--band-energy-window-s", "0.05",
        "--band-energy-step-s", "0.01",
        "--band-energy-normalize-per-trace",

        "--cave-extent-x-m", "140.5,160.0",
    ]

    print(f"\n{'=' * 80}")
    print(f"Running synthetic cave/no-cave comparison for {comp}")
    print(f"{'=' * 80}")
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
