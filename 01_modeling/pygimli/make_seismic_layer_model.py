#!/usr/bin/env python3
"""Create the layered seismic model and synthetic acquisition geometry.

The geological fills and the Vp colorbar deliberately share the *same*
Matplotlib colormap and Normalize object.  Consequently, every fill color is
exactly the color shown by the colorbar at that material's P-wave velocity.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Editable model and output settings
# -----------------------------------------------------------------------------
PROFILE_X_MIN_M = -225.0
PROFILE_X_MAX_M = 225.0
MODEL_BOTTOM_M = 50.0

LAYERS = (
    {
        "name": "Layer 1",
        "top": 0.0,
        "bottom": 10.0,
        "vp": 1600.0,
        "vs": 800.0,
        "density": 2050.0,
    },
    {
        "name": "Layer 2",
        "top": 10.0,
        "bottom": 35.0,
        "vp": 3600.0,
        "vs": 1800.0,
        "density": 2280.0,
    },
    {
        "name": "Layer 3",
        "top": 35.0,
        "bottom": MODEL_BOTTOM_M,
        "vp": 4700.0,
        "vs": 2600.0,
        "density": 2450.0,
    },
)

CAVE = {
    "name": "Water-filled cave",
    "center_x": 0.0,
    "width": 20.0,
    "top": 15.0,
    "bottom": 25.0,
    "vp": 1500.0,
    "vs": 0.0,
    "density": 1000.0,
}

RECEIVER_SPACING_M = 0.5
RECEIVER_X_MIN_M = -140.0
RECEIVER_X_MAX_M = 140.0
SOURCE_SPACING_M = 1.0
SOURCE_X_MIN_M = -150.0
SOURCE_X_MAX_M = 150.0

# Figure 3.3-style reversed inferno scale. Change to "magma_r" here if needed.
CMAP_NAME = "inferno_r"
VP_MIN = 604.0
VP_MAX = 5153.0
COLORBAR_TICKS = (604, 1741, 2879, 4016, 5153)

OUTPUT_STEM = "figure_3_1_seismic_layer_model"
FIGURE_SIZE_IN = (11.0, 5.8)
PNG_DPI = 400


def contrasting_text_color(rgba: tuple[float, float, float, float]) -> str:
    """Return black or white for readable text over an RGBA background."""
    r, g, b, _ = rgba
    # Relative luminance in sRGB (sufficient for choosing annotation color).
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.53 else "white"


def property_label(name: str, vp: float, vs: float, density: float) -> str:
    """Format a compact two-line material label."""
    return (
        rf"$\bf{{{name.replace(' ', '\\ ')}}}$" + "\n"
        rf"$V_P$ = {vp:,.0f} m/s   $V_S$ = {vs:,.0f} m/s   "
        rf"$\rho$ = {density:,.0f} kg/m$^3$"
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # One shared colormap and one shared normalization for fills and colorbar.
    cmap = mpl.colormaps[CMAP_NAME]
    norm = mpl.colors.Normalize(vmin=VP_MIN, vmax=VP_MAX, clip=True)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, constrained_layout=True)

    # Draw the three solid-color layers using cmap(norm(Vp)).
    for layer in LAYERS:
        color = cmap(norm(layer["vp"]))
        ax.add_patch(
            Rectangle(
                (PROFILE_X_MIN_M, layer["top"]),
                PROFILE_X_MAX_M - PROFILE_X_MIN_M,
                layer["bottom"] - layer["top"],
                facecolor=color,
                edgecolor="none",
                zorder=1,
            )
        )

        # Put labels toward the left so Layer 2 text does not compete with cave.
        center_y = (layer["top"] + layer["bottom"]) / 2.0
        ax.text(
            PROFILE_X_MIN_M + 10.0,
            center_y,
            property_label(
                layer["name"], layer["vp"], layer["vs"], layer["density"]
            ),
            ha="left",
            va="center",
            color=contrasting_text_color(color),
            linespacing=1.25,
            zorder=4,
        )

    # Layer boundaries are structural lines, not an independent color encoding.
    for depth in (0.0, 10.0, 35.0, MODEL_BOTTOM_M):
        ax.hlines(
            depth,
            PROFILE_X_MIN_M,
            PROFILE_X_MAX_M,
            color="0.12",
            linewidth=1.1,
            zorder=3,
        )

    # Draw cave using the same cmap(norm(Vp)) expression as the layers.
    cave_left = CAVE["center_x"] - CAVE["width"] / 2.0
    cave_color = cmap(norm(CAVE["vp"]))
    ax.add_patch(
        Rectangle(
            (cave_left, CAVE["top"]),
            CAVE["width"],
            CAVE["bottom"] - CAVE["top"],
            facecolor=cave_color,
            edgecolor="0.12",
            linewidth=1.5,
            zorder=5,
        )
    )
    # Place the cave properties outside the cave so they remain readable at the
    # final report scale. The arrow points to the cave's right-hand edge.
    cave_text = (
        r"$\bf{Water-filled\ cave}$" + "\n"
        rf"$V_P$ = {CAVE['vp']:,.0f} m/s   "
        rf"$V_S$ = {CAVE['vs']:,.0f} m/s   "
        rf"$\rho$ = {CAVE['density']:,.0f} kg/m$^3$"
    )
    ax.annotate(
        cave_text,
        xy=(cave_left + CAVE["width"], (CAVE["top"] + CAVE["bottom"]) / 2.0),
        xytext=(32.0, 20.0),
        textcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "linewidth": 1.3,
            "shrinkA": 5,
            "shrinkB": 2,
        },
        ha="left",
        va="center",
        color="white",
        fontsize=9.5,
        linespacing=1.2,
        zorder=7,
    )

    # Actual synthetic locations. Small vertical offsets keep both rows legible.
    receiver_x = np.arange(
        RECEIVER_X_MIN_M,
        RECEIVER_X_MAX_M + 0.5 * RECEIVER_SPACING_M,
        RECEIVER_SPACING_M,
    )
    source_x = np.arange(
        SOURCE_X_MIN_M,
        SOURCE_X_MAX_M + 0.5 * SOURCE_SPACING_M,
        SOURCE_SPACING_M,
    )
    ax.scatter(
        receiver_x,
        np.full_like(receiver_x, -0.8),
        marker="v",
        s=9,
        facecolor="#1677b8",
        edgecolor="none",
        clip_on=False,
        rasterized=True,
        zorder=8,
    )
    ax.scatter(
        source_x,
        np.full_like(source_x, -2.2),
        marker="*",
        s=13,
        facecolor="#c83737",
        edgecolor="none",
        clip_on=False,
        rasterized=True,
        zorder=9,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="none",
            markerfacecolor="#1677b8",
            markeredgecolor="none",
            markersize=6,
            label=f"Receivers ({RECEIVER_SPACING_M:g} m spacing)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#c83737",
            markeredgecolor="none",
            markersize=8,
            label=f"Sources ({SOURCE_SPACING_M:g} m spacing)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
        borderaxespad=0.0,
    )

    ax.set_xlim(PROFILE_X_MIN_M, PROFILE_X_MAX_M)
    ax.set_ylim(MODEL_BOTTOM_M, -3.5)  # Depth increases downward.
    ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xticks(np.arange(-200, 201, 50))
    ax.set_yticks(np.arange(0, MODEL_BOTTOM_M + 1, 10))
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        orientation="vertical",
        ticks=COLORBAR_TICKS,
        fraction=0.035,
        pad=0.025,
        aspect=24,
    )
    cbar.set_label(r"P-wave velocity, $V_P$ (m/s)")
    cbar.ax.set_yticklabels([f"{tick:d}" for tick in COLORBAR_TICKS])

    output_dir = Path(__file__).resolve().parent
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
