# segy_tools

Utilities for active-source seismic data used in the karst geophysics project.

## Package convention

NumPy gather arrays are shaped as:

```python
data.shape == (n_traces, n_samples)
```

ObsPy `Stream` objects are used for SEG-Y, SU, and MiniSEED workflows.

## Module layout

- `io.py` — file I/O for SEG-Y, SU, MiniSEED and basic ObsPy wrappers.
- `headers.py` — SEG-Y trace-header creation and coordinate scaling helpers.
- `gather.py` — Stream/array gather conversion, SEG-Y geometry extraction, gather plotting wrappers, and gather differencing.
- `processing.py` — time-domain trace/gather processing: demeaning, normalization, clipping, bandpass filtering, AGC, time gain.
- `spectral.py` — FFT spectra, frequency-vs-offset products, Charlie MST transform, and f-k filtering/spectra.
- `diffraction.py` — diffraction/NMO-style hyperbola scanning and velocity-grid diagnostics.
- `wavelets.py` — source wavelets and tapers such as Ricker and Gaussian functions.
- `plotting.py` — low-level plotting functions for wiggle, image, difference, and source-spectrum displays.
- `picking.py` — simple picking tools and Charlie-style interactive picker.
- `legacy_charlie.py` — support for Charlie Breithaupt's raw binary/header field format.
- `workflows.py` — higher-level workflows such as real/synthetic overlay plots.

## Refactor decision

The functions from `seismic_gather_utils_refactored.py` were not placed mostly in `gather.py`. Only functions that convert, write, compare, or plot whole gathers belong there. Gain, filtering, wavelets, f-k analysis, and diffraction scans are kept in dedicated modules so they can be reused for SEG-Y, SU, SEG-2/SEG-D conversions, and SPECfEM-derived gathers without making `gather.py` too large.
