# specfem_tools

Utilities for reading, converting, visualizing, and exporting SPECFEM2D models for the Karst Geophysics project.

This package provides a lightweight interface between SPECFEM2D model outputs and the Python seismic ecosystem (ObsPy, SEG-Y, Seismic Unix), while preserving survey geometry wherever possible.

## Modules

| Module | Purpose |
|---------|---------|
| `config.py` | Configuration classes and default export settings. |
| `io.py` | Core input/output routines for reading SPECFEM outputs, converting them to ObsPy Streams, exporting SEG-Y and Seismic Unix files, and generating model products. |
| `model.py` | Utilities for locating, identifying, and working with SPECFEM model directories and metadata. |
| `movie.py` | Generation of wavefield animations and movie products from SPECFEM outputs. |

## Features

The package includes utilities for:

- discovering SPECFEM2D `OUTPUT_FILES` directories;
- reading SPECFEM2D SEM ASCII gathers;
- reading SPECFEM2D Seismic Unix (`*.su`) output;
- converting SPECFEM shot gathers to ObsPy `Stream` objects;
- exporting SPECFEM gathers to SEG-Y and Seismic Unix;
- writing SEG-Y trace headers compatible with downstream processing packages;
- preserving both regular and irregular receiver geometries;
- preserving source and receiver coordinates, including topography (x and elevation);
- generating publication-quality SEG-Y products;
- generating quick-look wiggle plots;
- generating wavefield movies and animations;
- batch processing multiple SPECFEM models;
- comparing synthetic models and exported gathers.

## Geometry support

`specfem_tools` supports both traditional regularly sampled receiver arrays and arbitrary acquisition geometries.

Receiver and source geometry may be specified explicitly, allowing SEG-Y exports to preserve:

- irregular receiver spacing;
- irregular shot spacing;
- receiver elevations;
- source elevations;
- topographic profiles.

This enables direct comparison between SPECFEM simulations and real field surveys without resampling receiver locations.

## Package scope

`specfem_tools` is intended only for functionality that depends on SPECFEM2D.

Generic seismic utilities should instead reside in `segy_tools`, including:

- SEG-Y and Seismic Unix utilities;
- filtering;
- plotting;
- frequency-domain analysis;
- diffraction imaging;
- generic ObsPy utilities.

## Recommended imports

```python
from specfem_tools.io import (
    SpecfemExportConfig,
    find_specfem_model_outputs,
    model_number_from_name,
    model_name_from_output_dir,
    load_specfem_gather,
    write_model_products,
    batch_write_model_products,
    convert_sem_output_to_segy,
    convert_su_shot_to_segy,
    plot_segy_file,
    plot_su_directory,
    plot_model_difference_from_segy,
)
```

## Design philosophy

The package is organized around a single I/O interface (`io.py`) that handles:

```
SPECFEM2D outputs
        │
        ▼
    ObsPy Stream
        │
        ├── SEG-Y
        ├── Seismic Unix
        ├── plots
        ├── movies
        └── downstream analysis
```

All geometry handling is centralized within `io.py`, ensuring consistent treatment of source and receiver locations across all export formats.
