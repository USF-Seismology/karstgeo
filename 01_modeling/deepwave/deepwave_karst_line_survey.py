# deepwave_karst_line_survey.py

from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import deepwave
from deepwave import scalar

from obspy import Stream, Trace, UTCDateTime


# ============================================================
# 0. Output and CPU settings
# ============================================================

OUTDIR = Path("deepwave_karst_line_output")
OUTDIR.mkdir(exist_ok=True)

torch.set_num_threads(os.cpu_count())
device = torch.device("cpu")

print(f"Using device: {device}")
print(f"PyTorch threads: {torch.get_num_threads()}")


# ============================================================
# 1. Survey design
# ============================================================

# Fixed SmartSolo line
n_smartsolo = 36
smartsolo_spacing_m = 4.0
smartsolo_x_m = np.arange(n_smartsolo) * smartsolo_spacing_m
smartsolo_z_m = np.ones_like(smartsolo_x_m) * 1.0

line_start_m = smartsolo_x_m.min()
line_end_m = smartsolo_x_m.max()
line_centre_m = 0.5 * (line_start_m + line_end_m)

print(f"SmartSolo line: {line_start_m:.1f} to {line_end_m:.1f} m")
print(f"SmartSolo spacing: {smartsolo_spacing_m:.1f} m")
print(f"Line length: {line_end_m - line_start_m:.1f} m")


# Shot design:
# - off-end shots for refraction
# - shots at every SmartSolo node
# - far off-end shots
shot_x_m = np.unique(
    np.concatenate(
        [
            np.array([-20.0, -10.0]),
            smartsolo_x_m,
            np.array([150.0, 160.0]),
        ]
    )
)

shot_z_m = np.ones_like(shot_x_m) * 1.0

print(f"Number of shots: {len(shot_x_m)}")


# Optional rolling cabled spreads, 48 geophones at 1 m spacing.
# These are not used in the main SmartSolo synthetic output below,
# but are plotted and saved as a suggested field layout.
n_cabled = 48
cabled_spacing_m = 1.0
cabled_spread_length_m = (n_cabled - 1) * cabled_spacing_m

cabled_spread_starts_m = np.array([0.0, 23.0, 46.0, 69.0, 93.0])
cabled_spreads = []

for start in cabled_spread_starts_m:
    x = start + np.arange(n_cabled) * cabled_spacing_m
    x = x[x <= line_end_m]
    cabled_spreads.append(x)


# ============================================================
# 2. Deepwave model settings
# ============================================================

# This is the next step up from the quick test.
dx = 0.5
dt = 0.00025
nt = 1800             # 0.45 s total
freq = 80.0           # hammer-like dominant frequency

accuracy = 4
pml_width = 20

# Computational domain.
# Include room for off-end shots.
x_min_m = -30.0
x_max_m = 170.0
z_min_m = 0.0
z_max_m = 50.0

x = np.arange(x_min_m, x_max_m + dx, dx)
z = np.arange(z_min_m, z_max_m + dx, dx)

nx = len(x)
nz = len(z)

print(f"Model size: nz={nz}, nx={nx}, cells={nz * nx:,}")


def x_to_ix(x_m):
    return int(round((x_m - x_min_m) / dx))


def z_to_iz(z_m):
    return int(round((z_m - z_min_m) / dx))


# ============================================================
# 3. Velocity model
# ============================================================

v_np = np.ones((nz, nx), dtype=np.float32) * 2200.0

# Simple near-surface layering
Z, X = np.meshgrid(z, x, indexing="ij")

v_np[Z < 3.0] = 600.0
v_np[(Z >= 3.0) & (Z < 8.0)] = 1200.0
v_np[Z >= 8.0] = 2500.0

# Mild vertical gradient in limestone
v_np += (Z.astype(np.float32) * 8.0)

# Cave geometry:
# 6 m diameter, top at 15 m, so centre is 18 m.
cave_diameter_m = 6.0
cave_radius_m = cave_diameter_m / 2.0
cave_x_m = line_centre_m
cave_top_m = 15.0
cave_centre_z_m = cave_top_m + cave_radius_m

cave_mask = (
    (X - cave_x_m) ** 2
    + (Z - cave_centre_z_m) ** 2
    <= cave_radius_m ** 2
)

# Simplified scalar/acoustic cavity approximation.
# This is not a full elastic air-filled cavity model.
v_np[cave_mask] = 330.0

v = torch.tensor(v_np, dtype=torch.float32, device=device)


# ============================================================
# 4. Plot survey geometry and velocity model
# ============================================================

plt.figure(figsize=(12, 5))
plt.imshow(
    v_np,
    extent=[x.min(), x.max(), z.max(), z.min()],
    aspect="auto",
)
plt.colorbar(label="Velocity (m/s)")

plt.scatter(smartsolo_x_m, smartsolo_z_m, s=20, label="Fixed SmartSolo nodes")
plt.scatter(shot_x_m, shot_z_m, marker="*", s=80, label="Shot points")

for i, spread_x in enumerate(cabled_spreads):
    plt.scatter(
        spread_x,
        np.ones_like(spread_x) * (2.0 + 0.25 * i),
        s=8,
        alpha=0.7,
        label="Rolling 48-ch cabled spreads" if i == 0 else None,
    )

theta = np.linspace(0, 2 * np.pi, 200)
plt.plot(
    cave_x_m + cave_radius_m * np.cos(theta),
    cave_centre_z_m + cave_radius_m * np.sin(theta),
    linewidth=2,
    label="6 m cave target",
)

plt.xlabel("Distance along line (m)")
plt.ylabel("Depth (m)")
plt.title("Karst/cave single-line survey geometry")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTDIR / "survey_geometry_velocity_model.png", dpi=200)


# ============================================================
# 5. Source wavelet
# ============================================================

peak_time = 1.5 / freq

source_wavelet = deepwave.wavelets.ricker(
    freq,
    nt,
    dt,
    peak_time,
).reshape(1, 1, -1).to(device)


# ============================================================
# 6. Receiver locations: fixed SmartSolo baseline
# ============================================================

receiver_locations_np = np.column_stack(
    [
        [z_to_iz(zv) for zv in smartsolo_z_m],
        [x_to_ix(xv) for xv in smartsolo_x_m],
    ]
)

receiver_locations = torch.tensor(
    receiver_locations_np[None, :, :],
    dtype=torch.long,
    device=device,
)

n_receivers = receiver_locations.shape[1]


# ============================================================
# 7. Run shots one at a time and write MiniSEED
# ============================================================

starttime0 = UTCDateTime(2026, 1, 1)

all_shot_data = []

for i_shot, sx in enumerate(shot_x_m):
    print(f"Running shot {i_shot + 1:03d}/{len(shot_x_m):03d}: x={sx:.1f} m")

    source_locations = torch.tensor(
        [[[z_to_iz(shot_z_m[i_shot]), x_to_ix(sx)]]],
        dtype=torch.long,
        device=device,
    )

    out = scalar(
        v,
        grid_spacing=dx,
        dt=dt,
        source_amplitudes=source_wavelet,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=freq,
    )

    receiver_data = out[-1].detach().cpu().numpy()[0]
    all_shot_data.append(receiver_data.astype(np.float32))

    # Save NumPy shot gather
    np.save(
        OUTDIR / f"shot_{i_shot:03d}_x_{sx:07.2f}m_smartsolo.npy",
        receiver_data.astype(np.float32),
    )

    # Convert this shot to MiniSEED
    st = Stream()
    shot_starttime = starttime0 + i_shot * 10.0

    for i_rec, trace_data in enumerate(receiver_data):
        tr = Trace(data=trace_data.astype(np.float32))

        tr.stats.network = "SY"
        tr.stats.station = f"S{i_rec:03d}"
        tr.stats.location = f"{i_shot:02d}"
        tr.stats.channel = "BHZ"
        tr.stats.starttime = shot_starttime
        tr.stats.delta = dt

        tr.stats.coordinates = {
            "x_m": float(smartsolo_x_m[i_rec]),
            "z_m": float(smartsolo_z_m[i_rec]),
        }

        tr.stats.sac = {
            "dist": float(abs(smartsolo_x_m[i_rec] - sx)),
            "user0": float(smartsolo_x_m[i_rec]),
            "user1": float(smartsolo_z_m[i_rec]),
            "user2": float(sx),
            "user3": float(shot_z_m[i_shot]),
            "kevnm": f"SHOT{i_shot:03d}",
        }

        st.append(tr)

    mseed_path = OUTDIR / f"shot_{i_shot:03d}_x_{sx:07.2f}m_smartsolo.mseed"
    st.write(str(mseed_path), format="MSEED")


all_shot_data = np.stack(all_shot_data, axis=0)
np.save(OUTDIR / "all_shots_smartsolo.npy", all_shot_data)
np.save(OUTDIR / "shot_x_m.npy", shot_x_m)
np.save(OUTDIR / "smartsolo_x_m.npy", smartsolo_x_m)

print(f"All shot data shape: {all_shot_data.shape}")
print(f"Saved output to: {OUTDIR}")


# ============================================================
# 8. Plot example shot gathers
# ============================================================

time = np.arange(nt) * dt

def plot_shot_gather(shot_index, outfile):
    data = all_shot_data[shot_index].copy()
    data /= np.max(np.abs(data), axis=1, keepdims=True) + 1e-12

    sx = shot_x_m[shot_index]

    plt.figure(figsize=(11, 7))
    scale = 1.5

    for i_rec in range(n_receivers):
        plt.plot(
            smartsolo_x_m[i_rec] + scale * data[i_rec],
            time,
            linewidth=0.7,
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Receiver position along line (m)")
    plt.ylabel("Time (s)")
    plt.title(f"Synthetic SmartSolo shot gather: shot x = {sx:.1f} m")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


# Left off-end, centre, right off-end
plot_shot_gather(0, OUTDIR / "shot_gather_left_off_end.png")
centre_shot_index = int(np.argmin(np.abs(shot_x_m - cave_x_m)))
plot_shot_gather(centre_shot_index, OUTDIR / "shot_gather_centre.png")
plot_shot_gather(len(shot_x_m) - 1, OUTDIR / "shot_gather_right_off_end.png")


# ============================================================
# 9. Plot common receiver gather
# ============================================================

centre_receiver_index = int(np.argmin(np.abs(smartsolo_x_m - cave_x_m)))

crg = all_shot_data[:, centre_receiver_index, :].copy()
crg /= np.max(np.abs(crg), axis=1, keepdims=True) + 1e-12

plt.figure(figsize=(11, 7))
scale = 1.5

for i_shot, sx in enumerate(shot_x_m):
    plt.plot(
        sx + scale * crg[i_shot],
        time,
        linewidth=0.7,
    )

plt.gca().invert_yaxis()
plt.xlabel("Shot position along line (m)")
plt.ylabel("Time (s)")
plt.title(
    f"Common-receiver gather at SmartSolo x = "
    f"{smartsolo_x_m[centre_receiver_index]:.1f} m"
)
plt.tight_layout()
plt.savefig(OUTDIR / "common_receiver_gather_centre_receiver.png", dpi=200)
plt.close()


# ============================================================
# 10. Write survey table
# ============================================================

with open(OUTDIR / "survey_design_summary.txt", "w") as f:
    f.write("Karst/cave single-line survey design\n")
    f.write("====================================\n\n")
    f.write(f"SmartSolo nodes: {n_smartsolo}\n")
    f.write(f"SmartSolo spacing: {smartsolo_spacing_m:.1f} m\n")
    f.write(f"SmartSolo line: {line_start_m:.1f} to {line_end_m:.1f} m\n")
    f.write(f"SmartSolo line length: {line_end_m - line_start_m:.1f} m\n\n")

    f.write("Cave target:\n")
    f.write(f"  diameter: {cave_diameter_m:.1f} m\n")
    f.write(f"  top depth: {cave_top_m:.1f} m\n")
    f.write(f"  centre depth: {cave_centre_z_m:.1f} m\n")
    f.write(f"  line position: {cave_x_m:.1f} m\n\n")

    f.write("Shot positions, metres:\n")
    f.write(", ".join([f"{sx:.1f}" for sx in shot_x_m]))
    f.write("\n\n")

    f.write("Suggested rolling cabled spreads:\n")
    for i, spread_x in enumerate(cabled_spreads):
        f.write(
            f"  Spread {i + 1}: "
            f"{spread_x.min():.1f} to {spread_x.max():.1f} m, "
            f"{len(spread_x)} channels at {cabled_spacing_m:.1f} m spacing\n"
        )

print("Done.")
