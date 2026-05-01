#!/usr/bin/env python3
"""
deepwave_karst_model.py

Config-driven Deepwave scalar/acoustic forward model for a single-line
karst / water-filled cave seismic survey.

Outputs:
  - per-shot MiniSEED files
  - per-shot NumPy arrays
  - all-shots NumPy cube
  - survey geometry CSV files
  - velocity/density/impedance model figures
  - shot gather figures
  - wiggle gather figures
  - common receiver gather
  - first-arrival time estimate figure from a simple threshold picker

Notes:
  Deepwave scalar modelling uses a single acoustic wavespeed model. Density is
  included here for model documentation and impedance graphics, but not passed
  into deepwave.scalar(). For density-coupled amplitudes and elastic effects,
  use Deepwave elastic or SPECFEM2D later.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt

import deepwave
from deepwave import scalar
from obspy import Stream, Trace, UTCDateTime


# -----------------------------
# Configuration
# -----------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_torch(num_threads: int | None = None) -> torch.device:
    if num_threads is None:
        num_threads = os.cpu_count() or 1
    torch.set_num_threads(int(num_threads))
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    return device


# -----------------------------
# Geometry helpers
# -----------------------------

def make_axis(min_value: float, max_value: float, dx: float) -> np.ndarray:
    n = int(round((max_value - min_value) / dx)) + 1
    return min_value + np.arange(n) * dx


def coord_to_index(value: float, minimum: float, dx: float) -> int:
    return int(round((value - minimum) / dx))


def create_survey_geometry(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    survey = cfg["survey"]

    n_nodes = int(survey["smartsolo"]["n_nodes"])
    spacing = float(survey["smartsolo"]["spacing_m"])
    start = float(survey["smartsolo"].get("start_m", 0.0))
    smartsolo_x = start + np.arange(n_nodes) * spacing

    shot_cfg = survey["shots"]
    shots = []

    if shot_cfg.get("include_node_shots", True):
        shots.extend(list(smartsolo_x))

    shots.extend(shot_cfg.get("extra_shots_m", []))

    if "regular_spacing_m" in shot_cfg and shot_cfg["regular_spacing_m"] is not None:
        x0 = float(shot_cfg.get("regular_start_m", smartsolo_x.min()))
        x1 = float(shot_cfg.get("regular_end_m", smartsolo_x.max()))
        ds = float(shot_cfg["regular_spacing_m"])
        shots.extend(list(np.arange(x0, x1 + 0.001, ds)))

    shot_x = np.unique(np.asarray(shots, dtype=float))

    # Optional rolling cabled spreads
    cabled_spreads: list[np.ndarray] = []
    cabled = survey.get("cabled", {})
    if cabled.get("enabled", True):
        n_ch = int(cabled.get("n_channels", 48))
        cabled_spacing = float(cabled.get("spacing_m", 1.0))
        for start in cabled.get("spread_starts_m", []):
            spread = float(start) + np.arange(n_ch) * cabled_spacing
            cabled_spreads.append(spread)

    return smartsolo_x, shot_x, cabled_spreads


# -----------------------------
# Model building
# -----------------------------

def build_layered_model(
    cfg: dict[str, Any],
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Build Vp, density, and acoustic impedance models.

    Supports cave geometries:
      - circle
      - rectangle

    Notes
    -----
    deepwave.scalar() uses Vp only. Density is included here for
    graphics, impedance estimates, and documentation of the model.
    """
    model_cfg = cfg["earth_model"]
    cave_cfg = cfg["cave"]

    X, Z = np.meshgrid(x, z, indexing="xy")

    vp = np.ones_like(X, dtype=np.float32) * float(model_cfg["background"]["vp_m_s"])
    rho = np.ones_like(X, dtype=np.float32) * float(model_cfg["background"]["density_kg_m3"])

    for layer in model_cfg["layers"]:
        z_min = float(layer["z_min_m"])
        z_max = float(layer["z_max_m"])
        mask = (Z >= z_min) & (Z < z_max)

        vp[mask] = float(layer["vp_m_s"])
        rho[mask] = float(layer["density_kg_m3"])

    if model_cfg.get("gradient", {}).get("enabled", False):
        g = model_cfg["gradient"]
        z0 = float(g.get("z0_m", 0.0))
        grad = float(g.get("vp_gradient_m_s_per_m", 0.0))

        mask = Z >= z0
        vp[mask] += (Z[mask] - z0) * grad

    cave_geom = cave_cfg.get("geometry", "circle").lower()

    if cave_geom == "circle":
        diameter = float(cave_cfg["diameter_m"])
        radius = diameter / 2.0
        centre_x = float(cave_cfg["centre_x_m"])
        centre_z = float(cave_cfg["centre_depth_m"])

        cave_mask = (
            (X - centre_x) ** 2
            + (Z - centre_z) ** 2
            <= radius ** 2
        )

        cave_meta = {
            "geometry": "circle",
            "centre_x_m": centre_x,
            "centre_depth_m": centre_z,
            "diameter_m": diameter,
            "radius_m": radius,
            "top_depth_m": centre_z - radius,
            "bottom_depth_m": centre_z + radius,
        }

    elif cave_geom == "rectangle":
        x_min = float(cave_cfg["x_min_m"])
        x_max = float(cave_cfg["x_max_m"])
        z_min = float(cave_cfg["z_min_m"])
        z_max = float(cave_cfg["z_max_m"])

        cave_mask = (
            (X >= x_min) & (X <= x_max)
            & (Z >= z_min) & (Z <= z_max)
        )

        cave_meta = {
            "geometry": "rectangle",
            "x_min_m": x_min,
            "x_max_m": x_max,
            "z_min_m": z_min,
            "z_max_m": z_max,
            "centre_x_m": 0.5 * (x_min + x_max),
            "centre_depth_m": 0.5 * (z_min + z_max),
            "width_m": x_max - x_min,
            "height_m": z_max - z_min,
            "diameter_m": max(x_max - x_min, z_max - z_min),
            "radius_m": 0.5 * max(x_max - x_min, z_max - z_min),
            "top_depth_m": z_min,
            "bottom_depth_m": z_max,
        }

    else:
        raise ValueError(
            f"Unsupported cave geometry '{cave_geom}'. "
            "Use 'circle' or 'rectangle'."
        )

    # Water-filled cave properties.
    # For scalar Deepwave, only Vp affects propagation.
    vp[cave_mask] = float(cave_cfg["water"]["vp_m_s"])
    rho[cave_mask] = float(cave_cfg["water"]["density_kg_m3"])

    impedance = vp * rho

    return (
        vp.astype(np.float32),
        rho.astype(np.float32),
        impedance.astype(np.float32),
        cave_meta,
    )

# -----------------------------
# Modelling
# -----------------------------

def make_receiver_locations(
    receiver_x: np.ndarray,
    receiver_z_m: float,
    x_min: float,
    z_min: float,
    dx: float,
    device: torch.device,
) -> torch.Tensor:
    locs = np.column_stack([
        [coord_to_index(receiver_z_m, z_min, dx) for _ in receiver_x],
        [coord_to_index(xv, x_min, dx) for xv in receiver_x],
    ])
    return torch.tensor(locs[None, :, :], dtype=torch.long, device=device)


def run_one_shot(
    vp: np.ndarray,
    cfg: dict[str, Any],
    shot_x_m: float,
    receiver_x_m: np.ndarray,
    x_min_m: float,
    z_min_m: float,
    device: torch.device,
) -> np.ndarray:
    mcfg = cfg["deepwave"]
    dx = float(mcfg["dx_m"])
    dt = float(mcfg["dt_s"])
    nt = int(mcfg["nt"])
    freq = float(mcfg["source_frequency_hz"])

    source_z_m = float(cfg["survey"]["shots"].get("source_depth_m", 1.0))
    receiver_z_m = float(cfg["survey"]["smartsolo"].get("receiver_depth_m", 1.0))

    receiver_locations = make_receiver_locations(
        receiver_x_m, receiver_z_m, x_min_m, z_min_m, dx, device
    )

    source_locations = torch.tensor(
        [[[coord_to_index(source_z_m, z_min_m, dx),
           coord_to_index(shot_x_m, x_min_m, dx)]]],
        dtype=torch.long,
        device=device,
    )

    wavelet = deepwave.wavelets.ricker(
        freq,
        nt,
        dt,
        1.5 / freq,
    ).reshape(1, 1, -1).to(device)

    vp_t = torch.tensor(vp, dtype=torch.float32, device=device)

    out = scalar(
        vp_t,
        grid_spacing=dx,
        dt=dt,
        source_amplitudes=wavelet,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=int(mcfg.get("accuracy", 4)),
        pml_width=int(mcfg.get("pml_width", 20)),
        pml_freq=freq,
    )

    return out[-1].detach().cpu().numpy()[0].astype(np.float32)


# -----------------------------
# Output
# -----------------------------

def write_mseed(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    shot_x_m: float,
    shot_index: int,
    cfg: dict[str, Any],
    outdir: Path,
) -> None:
    dt = float(cfg["deepwave"]["dt_s"])
    start = UTCDateTime(cfg["output"].get("starttime", "2026-01-01T00:00:00")) + shot_index * 10.0
    network = cfg["output"].get("network", "SY")
    channel = cfg["output"].get("channel", "BHZ")

    st = Stream()
    for i, y in enumerate(data):
        tr = Trace(data=y.astype(np.float32))
        tr.stats.network = network
        tr.stats.station = f"S{i:03d}"
        tr.stats.location = f"{shot_index % 100:02d}"
        tr.stats.channel = channel
        tr.stats.starttime = start
        tr.stats.delta = dt
        tr.stats.sac = {
            "dist": float(abs(receiver_x_m[i] - shot_x_m)),
            "user0": float(receiver_x_m[i]),
            "user1": float(cfg["survey"]["smartsolo"].get("receiver_depth_m", 1.0)),
            "user2": float(shot_x_m),
            "user3": float(cfg["survey"]["shots"].get("source_depth_m", 1.0)),
        }
        st.append(tr)

    st.write(str(outdir / f"shot_{shot_index:03d}_x_{shot_x_m:08.2f}m.mseed"), format="MSEED")


def write_csv_geometry(
    smartsolo_x: np.ndarray,
    shot_x: np.ndarray,
    cabled_spreads: list[np.ndarray],
    outdir: Path,
) -> None:
    with open(outdir / "smartsolo_nodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "x_m", "z_m"])
        for i, x in enumerate(smartsolo_x):
            writer.writerow([f"S{i:03d}", float(x), 1.0])

    with open(outdir / "shot_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shot", "x_m", "z_m"])
        for i, x in enumerate(shot_x):
            writer.writerow([f"SHOT{i:03d}", float(x), 1.0])

    with open(outdir / "cabled_spreads.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spread", "channel", "x_m", "z_m"])
        for si, spread in enumerate(cabled_spreads):
            for ci, x in enumerate(spread):
                writer.writerow([si, ci, float(x), 1.0])


# -----------------------------
# Graphics
# -----------------------------

def save_model_image(
    arr: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    smartsolo_x: np.ndarray,
    shot_x: np.ndarray,
    cabled_spreads: list[np.ndarray],
    cave_meta: dict[str, float],
    outpath: Path,
    title: str,
    cbar_label: str,
) -> None:
    plt.figure(figsize=(13, 5))
    plt.imshow(arr, extent=[x.min(), x.max(), z.max(), z.min()], aspect="auto")
    plt.colorbar(label=cbar_label)

    plt.scatter(smartsolo_x, np.ones_like(smartsolo_x), s=22, label="Fixed SmartSolo nodes")
    plt.scatter(shot_x, np.ones_like(shot_x) * 1.7, marker="*", s=60, label="Shot points")

    for i, spread in enumerate(cabled_spreads):
        plt.scatter(
            spread,
            np.ones_like(spread) * (2.5 + i * 0.25),
            s=7,
            alpha=0.65,
            label="Rolling 48-channel cabled spreads" if i == 0 else None,
        )

    theta = np.linspace(0, 2 * np.pi, 300)
    plt.plot(
        cave_meta["centre_x_m"] + cave_meta["radius_m"] * np.cos(theta),
        cave_meta["centre_depth_m"] + cave_meta["radius_m"] * np.sin(theta),
        linewidth=2,
        label="Water-filled cave",
    )

    plt.xlabel("Distance along line (m)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_layer_profile(vp: np.ndarray, rho: np.ndarray, x: np.ndarray, z: np.ndarray, outpath: Path) -> None:
    ix = len(x) // 2
    plt.figure(figsize=(5, 7))
    plt.plot(vp[:, ix], z, label="Vp (m/s)")
    plt.xlabel("Vp (m/s)")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    ax2 = plt.gca().twiny()
    ax2.plot(rho[:, ix], z, linestyle="--", label="Density (kg/m³)")
    ax2.set_xlabel("Density (kg/m³)")
    plt.title("Central 1-D model profile")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def normalise_traces(data: np.ndarray) -> np.ndarray:
    return data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-12)


def save_wiggle_gather(
    data: np.ndarray,
    receiver_x: np.ndarray,
    dt: float,
    outpath: Path,
    title: str,
    scale: float = 1.5,
) -> None:
    t = np.arange(data.shape[1]) * dt
    d = normalise_traces(data.copy())

    plt.figure(figsize=(12, 8))
    for i in range(data.shape[0]):
        plt.plot(receiver_x[i] + scale * d[i], t, linewidth=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("Receiver position (m)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_image_gather(
    data: np.ndarray,
    receiver_x: np.ndarray,
    dt: float,
    outpath: Path,
    title: str,
) -> None:
    t = np.arange(data.shape[1]) * dt
    d = data.copy()
    clip = np.percentile(np.abs(d), 98)
    if clip == 0:
        clip = 1.0

    plt.figure(figsize=(12, 7))
    plt.imshow(
        d.T,
        extent=[receiver_x.min(), receiver_x.max(), t.max(), t.min()],
        aspect="auto",
        vmin=-clip,
        vmax=clip,
    )
    plt.colorbar(label="Synthetic amplitude")
    plt.xlabel("Receiver position (m)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_common_receiver_gather(
    all_data: np.ndarray,
    shot_x: np.ndarray,
    receiver_index: int,
    dt: float,
    outpath: Path,
) -> None:
    crg = all_data[:, receiver_index, :]
    t = np.arange(crg.shape[1]) * dt
    d = normalise_traces(crg.copy())

    plt.figure(figsize=(12, 8))
    scale = 1.5
    for i, sx in enumerate(shot_x):
        plt.plot(sx + scale * d[i], t, linewidth=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("Shot position (m)")
    plt.ylabel("Time (s)")
    plt.title(f"Common-receiver gather, receiver index {receiver_index}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def simple_first_arrivals(data: np.ndarray, dt: float, threshold_fraction: float = 0.08) -> np.ndarray:
    picks = np.full(data.shape[0], np.nan)
    for i, tr in enumerate(data):
        env = np.abs(tr)
        threshold = threshold_fraction * np.max(env)
        idx = np.where(env >= threshold)[0]
        if len(idx):
            picks[i] = idx[0] * dt
    return picks


def save_first_arrival_plot(
    data: np.ndarray,
    receiver_x: np.ndarray,
    shot_x: float,
    dt: float,
    outpath: Path,
) -> None:
    picks = simple_first_arrivals(data, dt)

    plt.figure(figsize=(9, 5))
    plt.plot(np.abs(receiver_x - shot_x), picks, marker="o")
    plt.xlabel("Offset (m)")
    plt.ylabel("Picked first arrival time (s)")
    plt.title(f"Simple threshold first-arrival picks, shot x={shot_x:.1f} m")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = setup_torch(cfg.get("compute", {}).get("num_threads"))

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    dx = float(cfg["deepwave"]["dx_m"])
    x = make_axis(float(cfg["domain"]["x_min_m"]), float(cfg["domain"]["x_max_m"]), dx)
    z = make_axis(float(cfg["domain"]["z_min_m"]), float(cfg["domain"]["z_max_m"]), dx)

    smartsolo_x, shot_x, cabled_spreads = create_survey_geometry(cfg)
    vp, rho, impedance, cave_meta = build_layered_model(cfg, x, z)

    print(f"Model grid: nz={len(z)}, nx={len(x)}, cells={len(z) * len(x):,}")
    print(f"SmartSolo nodes: {len(smartsolo_x)}, shots: {len(shot_x)}")
    print(f"Cave top depth: {cave_meta['top_depth_m']:.2f} m")
    print(f"Cave centre depth: {cave_meta['centre_depth_m']:.2f} m")
    print(f"Cave diameter: {cave_meta['diameter_m']:.2f} m")

    write_csv_geometry(smartsolo_x, shot_x, cabled_spreads, outdir)

    save_model_image(
        vp, x, z, smartsolo_x, shot_x, cabled_spreads, cave_meta,
        outdir / "model_vp_survey_geometry.png",
        "P-wave velocity model and survey geometry",
        "Vp (m/s)",
    )
    save_model_image(
        rho, x, z, smartsolo_x, shot_x, cabled_spreads, cave_meta,
        outdir / "model_density_survey_geometry.png",
        "Density model and survey geometry",
        "Density (kg/m³)",
    )
    save_model_image(
        impedance, x, z, smartsolo_x, shot_x, cabled_spreads, cave_meta,
        outdir / "model_impedance_survey_geometry.png",
        "Acoustic impedance model and survey geometry",
        "Vp × density",
    )
    save_layer_profile(vp, rho, x, z, outdir / "central_1d_model_profile.png")

    all_data = []

    for i, sx in enumerate(shot_x):
        print(f"Running shot {i + 1:03d}/{len(shot_x):03d}: x={sx:.2f} m")
        data = run_one_shot(vp, cfg, sx, smartsolo_x, x.min(), z.min(), device)
        all_data.append(data)

        np.save(outdir / f"shot_{i:03d}_x_{sx:08.2f}m.npy", data)

        if cfg["output"].get("write_mseed", True):
            write_mseed(data, smartsolo_x, sx, i, cfg, outdir)

    all_data_arr = np.stack(all_data, axis=0)
    np.save(outdir / "all_shots_smartsolo.npy", all_data_arr)
    np.save(outdir / "smartsolo_x_m.npy", smartsolo_x)
    np.save(outdir / "shot_x_m.npy", shot_x)

    dt = float(cfg["deepwave"]["dt_s"])

    # Representative gathers
    plot_indices = sorted(set([
        0,
        int(np.argmin(np.abs(shot_x - cave_meta["centre_x_m"]))),
        len(shot_x) - 1,
    ]))

    for idx in plot_indices:
        sx = shot_x[idx]
        save_wiggle_gather(
            all_data_arr[idx],
            smartsolo_x,
            dt,
            outdir / f"wiggle_shot_{idx:03d}_x_{sx:08.2f}m.png",
            f"Wiggle gather, shot x={sx:.1f} m",
        )
        save_image_gather(
            all_data_arr[idx],
            smartsolo_x,
            dt,
            outdir / f"image_shot_{idx:03d}_x_{sx:08.2f}m.png",
            f"Image gather, shot x={sx:.1f} m",
        )
        save_first_arrival_plot(
            all_data_arr[idx],
            smartsolo_x,
            sx,
            dt,
            outdir / f"first_arrivals_shot_{idx:03d}_x_{sx:08.2f}m.png",
        )

    centre_receiver = int(np.argmin(np.abs(smartsolo_x - cave_meta["centre_x_m"])))
    save_common_receiver_gather(
        all_data_arr,
        shot_x,
        centre_receiver,
        dt,
        outdir / "common_receiver_gather_near_cave_centre.png",
    )

    with open(outdir / "run_summary.txt", "w") as f:
        f.write("Deepwave karst survey model run summary\n")
        f.write("======================================\n\n")
        f.write(f"Grid spacing: {dx} m\n")
        f.write(f"Time step: {cfg['deepwave']['dt_s']} s\n")
        f.write(f"Samples: {cfg['deepwave']['nt']}\n")
        f.write(f"Source frequency: {cfg['deepwave']['source_frequency_hz']} Hz\n")
        f.write(f"SmartSolo nodes: {len(smartsolo_x)}\n")
        f.write(f"Shots: {len(shot_x)}\n\n")
        f.write("Cave:\n")
        for k, vmeta in cave_meta.items():
            f.write(f"  {k}: {vmeta}\n")
        f.write("\nNote: density is used for graphics/documentation only in scalar modelling.\n")

    print(f"Done. Outputs written to {outdir}")


if __name__ == "__main__":
    main()
