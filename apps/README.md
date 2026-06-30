# Shot Gather Applications

These applications provide interactive visualization, quality control, and manual picking of active-source seismic shot gathers.

The long-term design philosophy is that **all seismic data (nodal, Geode, streamer, SPECFEM synthetics, Deepwave synthetics, etc.) are first converted into geometry-aware SEG-Y**, after which all visualization, picking and processing is performed using the common `segy_tools` library.

---

# Applications

## segy_wiggle_picker.py

**Status:** New architecture (recommended)

This is the next-generation interactive shot gather viewer built directly on the `segy_tools` package.

Its purpose is to become the primary GUI for:

- viewing SEG-Y shot gathers
- overlaying multiple gathers
- manual picking
- automatic picking
- travel-time analysis
- tomography preparation

Unlike earlier applications, almost all seismic functionality lives inside `lib/segy_tools`, making this application primarily a graphical interface.

### Current capabilities

- Load SEG-Y
- Load SU
- Load MiniSEED
- Optional SQLite geometry patching
- Overlay two gathers
- Manual picking
- Save picks to CSV
- Interactive matplotlib display
- Geometry-aware plotting

### Planned capabilities

- Multiple simultaneous gathers
- Common-shot project manager
- Receiver binning
- Cross-correlation alignment
- Priority-based composite gathers
- Automatic first-break picking
- Travel-time curve display
- Velocity estimation
- FK filtering
- Acausal high-pass filtering
- Export to pyGIMLi
- Export to Refrainv
- Integrated QC tools

This application should eventually become the only GUI required for active-source seismic analysis.

---

## wiggle_picker.py

**Status:** Legacy application (still useful)

This was the original standalone shot gather picker developed before the `segy_tools` refactor.

It contains a large amount of functionality that has since been migrated into:

- `segy_tools.io`
- `segy_tools.gather`
- `segy_tools.plotting`
- `segy_tools.picking`
- `segy_tools.processing`

It remains useful because it already supports many interactive workflows.

### Features

- MiniSEED
- SEG-Y
- SU
- SEG-2
- Interactive wiggle plots
- Overlay mode
- Filtering
- Gain controls
- Time windows
- Positive/negative fill
- Manual picks
- Velocity line measurements
- CSV export

### Limitations

Internally this application still contains duplicated functionality that now belongs in `segy_tools`.

Over time this code should shrink until it simply becomes a GUI layer around the common library.

---

# Design philosophy

These applications should contain **very little seismic processing code.**

Instead they should call reusable routines in `lib/segy_tools`.

```
GUI
 │
 ▼
segy_tools
 ├── io.py
 ├── gather.py
 ├── commonshot.py
 ├── plotting.py
 ├── picking.py
 ├── processing.py
 ├── spectral.py
 ├── diffraction.py
 └── db.py
```

This separation allows the same algorithms to be used from:

- Jupyter notebooks
- command-line scripts
- automated processing pipelines
- graphical applications

without duplication.

---

# Long-term workflow

The intended workflow for all active-source seismic processing is:

```
Raw data
    │
    ▼
Convert to geometry-aware SEG-Y
    │
    ▼
Load into CommonShotProject
    │
    ├── receiver binning
    ├── common receiver identification
    ├── cross-correlation alignment
    ├── FK filtering
    ├── automatic picking
    ├── manual QC
    ├── travel-time curves
    └── composite gather generation
    │
    ▼
Export picks
    │
    ├── pyGIMLi
    ├── Refrainv
    ├── CSV
    └── future inversion packages
```

---

# Current development status

The core `segy_tools` refactor is largely complete.

Major modules now exist for:

- geometry-aware I/O
- gather manipulation
- plotting
- picking
- processing
- diffraction analysis
- spectral analysis
- database utilities
- multi-survey common-shot analysis

The current development focus is on:

1. finishing `commonshot.py`
2. correctly handling SEG-Y coordinate scaling
3. building the new GUI around `CommonShotProject`
4. integrating automatic picking and tomography export

Once complete, essentially all active-source processing will operate through a single geometry-aware workflow built on `segy_tools`.