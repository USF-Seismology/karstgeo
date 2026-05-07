from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .config import load_config, figures_dir, receiver_x


def parse_interfaces_file(path):
    path = Path(path).expanduser()
    lines = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    idx = 0
    n_interfaces = int(lines[idx].split()[0]); idx += 1
    interfaces = []
    for _ in range(n_interfaces):
        n_points = int(lines[idx].split()[0]); idx += 1
        xs, zs = [], []
        for _ in range(n_points):
            parts = lines[idx].split(); idx += 1
            xs.append(float(parts[0])); zs.append(float(parts[1]))
        interfaces.append((np.asarray(xs), np.asarray(zs)))
    nelem_layers = []
    while idx < len(lines):
        nelem_layers.append(int(lines[idx].split()[0])); idx += 1
    return interfaces, nelem_layers


def plot_interface_geometry(interfaces, nelem_layers=None, cave=None, source_x=None, receiver_min=None, receiver_max=None, outfile=None, dpi=160):
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (x, z) in enumerate(interfaces):
        ax.plot(x, z, linewidth=2, label=f"interface {i+1}")
    if cave:
        rect = patches.Rectangle((cave["x_min_m"], cave["z_min_m"]), cave["x_max_m"] - cave["x_min_m"], cave["z_max_m"] - cave["z_min_m"], alpha=0.25, label="known cave/void")
        ax.add_patch(rect)
    if source_x is not None:
        ax.axvline(source_x, linestyle="--", linewidth=1, label="example source")
    if receiver_min is not None and receiver_max is not None:
        ax.axvspan(receiver_min, receiver_max, alpha=0.08, label="receiver line")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_title("SPECFEM2D model geometry")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True); fig.savefig(outfile, dpi=dpi); plt.close(fig)
    return fig


def make_geometry_plot_from_config(config, outfile=None, dpi=None):
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    path = cfg.get("interfaces", {}).get("file")
    if not path:
        raise ValueError("Set interfaces.file in the YAML config.")
    interfaces, nelem = parse_interfaces_file(path)
    rx = receiver_x(cfg)
    if outfile is None:
        outfile = figures_dir(cfg) / "model_geometry.png"
    if dpi is None:
        dpi = cfg.get("plotting", {}).get("dpi", 160)
    fig = plot_interface_geometry(interfaces, nelem_layers=nelem, cave=cfg.get("cave"), source_x=cfg["survey"].get("first_shot_x_m"), receiver_min=float(rx.min()), receiver_max=float(rx.max()), outfile=outfile, dpi=dpi)
    return fig, Path(outfile)
