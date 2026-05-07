# SPECFEM / SEG-Y tools, split package

This version deliberately separates **SPECFEM-specific** code from **generic SEG-Y / ObsPy Stream** code.

## Package boundary

```text
specfem_tools/
    config.py       # YAML config, SPECFEM survey folders, receiver/shot x arrays
    io.py           # SPECFEM ASCII .semv/.semd/.sema readers
    model.py        # SPECFEM interfaces file + model geometry plots
    movie.py        # forward_image*.jpg -> movie
    converters.py   # SPECFEM ASCII/SU -> SEG-Y bridge

segy_tools/
    headers.py      # SEG-Y trace header creation and header forcing
    io.py           # read/write SEG-Y, SU, MiniSEED as ObsPy Streams
    gather.py       # Stream -> gather arrays; plot_*_from_stream; plot_*_from_segy
    plotting.py     # generic wiggle/image/difference/source-function plots
    picking.py      # generic first-arrival picking
```

Compatibility wrappers remain in `specfem_tools.core`, `specfem_tools.segy`, `specfem_tools.plotting`, etc., but new code should import from the split modules above.

## Preferred workflow

```text
SPECFEM .semv/.semd/.sema or SU
        ↓
specfem_tools.converters
        ↓
SEG-Y with useful headers
        ↓
segy_tools + ObsPy Stream workflows
```

Real survey data can enter directly at the SEG-Y/Stream stage.

## Notebook examples

```python
import obspy
from segy_tools.gather import plot_wiggle_gather_from_stream

st = obspy.read("Sarah_Mod17_SEGY/BXZ/shot_001_BXZ_semv.segy", format="SEGY", unpack_trace_headers=True)
fig = plot_wiggle_gather_from_stream(st, tmin=0, tmax=0.15, cave={"x_min_m": 145, "x_max_m": 155})
```

```python
from specfem_tools.converters import Geometry, Timing, convert_sem_output_to_segy

out = convert_sem_output_to_segy(
    input_dir="Mod17/OUTPUT_FILES",
    output_dir="Sarah_Mod17_SEGY",
    component="Z",
    extension="semv",
    shot_number=1,
    source_x_m=150.0,
    geom=Geometry(first_receiver_x_m=0.0, receiver_spacing_m=0.5),
    timing=Timing(dt_s=0.00002, t0_s=0.0),
)
```

## Install

```bash
pip install -e .
```
