#!/usr/bin/env python3
"""Generate a SPECFEM2D DATA/SOURCE file from a source-family CSV row."""
from pathlib import Path
import argparse
import pandas as pd


def fmt(x, default="0."):
    if pd.isna(x) or str(x)=="": return default
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", default="inputs/sources/source_families.csv")
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--xs", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="templates/SOURCE_ricker.template")
    args = ap.parse_args()
    df = pd.read_csv(args.source_csv)
    row = df[df.source_name == args.source_name]
    if row.empty:
        raise SystemExit(f"Unknown source_name: {args.source_name}")
    r = row.iloc[0]
    f0 = float(r.f0_hz)
    # A conservative default: 2/f0 puts the Ricker mostly after t=0; customize if needed.
    tshift = 2.0 / f0
    content = Path(args.template).read_text().format(
        xs=f"{args.xs:.6f}",
        source_type=int(r.source_type),
        f0=f"{f0:.6f}",
        tshift=f"{tshift:.6f}",
        anglesource=fmt(r.anglesource_deg, "180.0"),
        Mxx=fmt(r.Mxx, "1."),
        Mzz=fmt(r.Mzz, "1."),
        Mxz=fmt(r.Mxz, "0."),
        factor=fmt(r.factor, "1.d10"),
    )
    Path(args.out).write_text(content)

if __name__ == "__main__":
    main()
