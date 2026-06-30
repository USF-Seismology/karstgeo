# segy_tools

`segy_tools` is the geometry-aware seismic utility layer for the Karst Geophysics project.

The intended workflow is now:

```text
SPECFEM synthetics / Geode files / nodal gathers / stacked gathers
        ↓
geometry-aware SEG-Y files with source and receiver metadata in headers
        ↓
segy_tools plotting, processing, picking, alignment, and export tools
        ↓
RefraPick / Refrainv / pyGIMLi / project database / reports
```

The package should remain **generic** wherever possible. SPECFEM-specific logic belongs in `specfem_tools`; project database logic belongs in `db.py` or notebook/project-level helpers; plotting, picking, and processing should operate primarily on SEG-Y/ObsPy objects with receiver-position-aware geometry.

## Core conventions

Gather arrays use:

```python
data.shape == (n_traces, n_samples)
time_s.shape == (n_samples,)
receiver_x_m.shape == (n_traces,)
```

Coordinates are in meters. SEG-Y coordinates and elevations may be stored as scaled integers, but decoded values exposed by `segy_tools` should be in meters.

For 2-D profile work:

- `source_x_m` is the shot/source position along the transect.
- `receiver_x_m` is the receiver position along the transect.
- `offset_m = receiver_x_m - source_x_m` unless a downstream tool requires absolute offset.
- Receiver and source elevations should be stored as positive-up elevations when available.

## Current module layout

| Module | Status | Purpose |
|---|---|---|
| `io.py` | Core | Generic SEG-Y/SU/MiniSEED I/O, SEG-Y header creation, coordinate/elevation scaling, geometry extraction, array-to-Stream conversion, component filtering, and simple waveform preprocessing. This module should not depend on SQLite or project-specific databases. |
| `gather.py` | Core | Gather-array extraction, receiver-coordinate alignment, gather differencing, and deprecated plotting shims. Plotting functions should live in `plotting.py`. |
| `plotting.py` | Core | Generic wiggle/image/difference/source-spectrum/frequency-contour plotting for geometry-aware gathers and ObsPy Streams. Does not require picks. |
| `processing.py` | Core | Generic trace/gather processing: demeaning, normalization, clipping, bandpass filtering, AGC, and time gain. |
| `picking.py` | Core/emerging | Geometry-aware pick data structures, CSV helpers, Baer/AIC/AR-style automated picking, consensus picking, and pick-QC plots. Pick-dependent plots live here rather than in `plotting.py`. |
| `commonshot.py` | Experimental but important | Non-GUI engine for loading multiple SEG-Y gathers at the same/similar source position, binning common receivers, estimating inter-gather time shifts, building priority-based composite gathers, storing picks by source/receiver/phase, and exporting travel-time tables. |
| `db.py` | Project helper | SQLite/project-database geometry lookup helpers. Kept separate from `io.py` so generic SEG-Y code remains database-independent. |
| `diffraction.py` | Generic analysis | Diffraction/NMO-style hyperbola scanning and velocity-grid diagnostics. Updated to use receiver geometry where available. |
| `spectral.py` | Generic analysis | Spectra, frequency-vs-offset products, and f-k filtering/spectra. f-k routines require regular receiver spacing or explicit regularization. |
| `wavelets.py` | Generic source tools | Source wavelets and tapers such as Ricker/Gaussian functions. |
| `nodal_shotgather.py` | Compatibility shim | Historical nodal workflow module. Most generic picking/plotting functionality has moved into `picking.py` and `plotting.py`; project/SDS-specific workflows should remain outside core `segy_tools`. |

Historical modules such as `headers.py`, `charlie.py`, and `workflows.py` should either be removed or reduced to compatibility shims if anything still imports them.

## Recommended imports

```python
from segy_tools import io, gather, plotting, processing, picking
from segy_tools.commonshot import CommonShotProject
```

For notebooks run from the repository root, use:

```python
from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
LIB_DIR = REPO_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
```

Longer term, replace repeated notebook path setup with one project-level helper.

## SEG-Y geometry handling

`io.py` is the canonical place for SEG-Y header and geometry handling. It should provide and maintain:

- trace-header creation;
- source and receiver coordinate encoding/decoding;
- coordinate and elevation scalar handling;
- source/receiver elevation fields;
- geometry extraction from ObsPy SEG-Y/SU Streams;
- applying already-loaded geometry to Streams;
- writing geometry-aware SEG-Y files.

Database lookup is intentionally outside `io.py`. The intended split is:

```text
SQLite/project DB → pandas/DataFrame/mapping → segy_tools.io.apply_geometry_map()
```

not:

```text
segy_tools.io → SQLite
```

## Common-shot workflow status

`commonshot.py` is the foundation for the next-generation picking GUI and batch alignment tools. It is intended to support loading multiple shot gathers with the same or nearly the same source position, for example:

- wide nodal array;
- dense nodal array;
- T1 1-m refraction Geode gather;
- T1 2-m refraction Geode gather;
- T1 streamer gather;
- SPECFEM synthetic gather.

The intended common-shot workflow is:

```text
Load multiple SEG-Y gathers
        ↓
Verify compatible source_x_m
        ↓
Bin receivers by receiver_x_m tolerance, e.g. 0.10 m
        ↓
Estimate gather-wide time shifts from common-receiver cross-correlation
        ↓
Build a priority-based composite gather
        ↓
Pick by source_x_m + receiver_x_m + phase, not by trace number
        ↓
Export native picks / RefraPick-style tables / pyGIMLi travel-time tables
```

Current smoke testing shows that the engine now runs, but **coordinate scaling still needs verification**. In one test, decoded geometry looked like:

```text
receiver_x_m = 0.5, 2.5, 4.5, ...
source_x_m = 16250.0
```

where expected values were closer to profile coordinates such as receiver positions around tens to hundreds of meters and source positions around `162.5 m`. Before building the next GUI, verify and fix SEG-Y coordinate decoding/scaling so every gather reports:

```text
source_x_m ≈ real shot position in meters
receiver_x_m ≈ real receiver position in meters
offset_m = receiver_x_m - source_x_m
```

The current time-shift estimates from the smoke test also had low median correlations, so alignment results should not yet be trusted until geometry and processing windows are confirmed.

## Picking status

`picking.py` now contains both automated picking helpers and pick-QC plotting. This is deliberate: plots that require pick dictionaries, consensus pick tables, or AR/Baer/AIC pick metadata belong with picking rather than generic plotting.

The next target data model is a geometry-aware pick table keyed by:

```text
source_x_m, receiver_x_m, phase
```

rather than by gather label or trace index. Changing display priority in a composite gather should not change where picks are stored.

Planned export targets:

- native `segy_tools` CSV;
- RefraPick/Refrainv-friendly CSV;
- pyGIMLi travel-time table;
- optional SQLite/project database tables.

## GUI status

The old wiggle picker accumulated too many historical code paths. A new GUI should be built as a thin layer on top of `commonshot.py`, not by reimplementing loading, alignment, geometry, or pick-export logic.

The next GUI should provide:

- load multiple SEG-Y gathers;
- reorder gather display priority;
- set receiver bin tolerance;
- estimate/apply/reset time shifts;
- choose processing chain, e.g. acausal high-pass or f-k filtering;
- run autopicker from `picking.py`;
- display composite wiggle gather;
- display linked travel-time curves;
- save/export picks.

## Near-term TODO

1. Fix/verify SEG-Y coordinate scalar decoding for all exported Geode/nodal/SPECFEM files.
2. Add a small `commonshot.py` smoke-test script using real SEG-Y files and assert expected geometry ranges.
3. Improve common-receiver time-shift estimation using explicit windows and robust correlation thresholds.
4. Add travel-time curve plotting for common-shot pick tables.
5. Build the new GUI on top of `CommonShotProject` only after the engine is trustworthy.
6. Clean up legacy modules/imports once notebooks and apps use the new module boundaries.

## Design rule of thumb

- If it reads/writes seismic files or SEG-Y headers: `io.py`.
- If it converts Streams to gather arrays or aligns/differences gathers: `gather.py`.
- If it plots gathers without picks: `plotting.py`.
- If it creates, stores, QC-plots, or exports picks: `picking.py`.
- If it combines multiple gathers at the same shot position: `commonshot.py`.
- If it touches SQLite/project catalog tables: `db.py` or a project-local helper, not `io.py`.
- If it is SPECFEM-specific: `specfem_tools`, not `segy_tools`.
