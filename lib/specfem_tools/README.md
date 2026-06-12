# specfem_tools

SPECFEM2D-specific helper package for the Karst Geophysics project.

This package contains utilities for:

- reading SPECFEM2D SEM ASCII gathers;
- discovering `OUTPUT_FILES` directories;
- reading SPECFEM2D Seismic Unix (`*.su`) output;
- converting SPECFEM model gathers to ObsPy Streams with SEG-Y headers;
- exporting model outputs to SEG-Y;
- generating quick-look wiggle plots;
- comparing converted model gathers.

Generic SEG-Y, SU, filtering, plotting, diffraction, and spectral-analysis tools
belong in `segy_tools`, not here.

## Recommended imports

```python
from specfem_tools.output import (
    SpecfemExportConfig,
    find_specfem_model_outputs,
    write_model_products,
    batch_write_model_products,
)
```

A backward-compatible shim is provided:

```python
from specfem_tools.specfem_output_tools_refactored import SpecfemExportConfig
```

but new notebooks should prefer `specfem_tools.output`.
