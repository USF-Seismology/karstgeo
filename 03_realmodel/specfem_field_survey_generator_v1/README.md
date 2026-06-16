# SPECFEM2D Field-Survey Model Generation

This repository generates SPECFEM2D input folders for synthetic modelling of the karst seismic surveys. The goal is to make the modelling workflow reproducible, traceable, and directly comparable with the field data.

The key principle is:

> Do not hand-edit individual SPECFEM runs. Generate run directories from templates, field metadata, receiver/source geometry tables, and cave/no-cave model definitions.

---

## 1. Current workflow

The workflow is now:

```text
Excel field metadata
        ↓
scripts/build_geometry_library_from_excel.py
        ↓
inputs/surveys/
inputs/receiver_components/
inputs/receivers/
        ↓
generate_all_survey_run_suites.sh
        ↓
scripts/generate_specfem_runs.py
        ↓
runs/
```

The Excel workbooks remain the authoritative raw metadata. The generated CSV files under `inputs/` are the machine-readable geometry library used by the run generator.

---

## 2. Current repository layout

```text
specfem_field_survey_generator_v1/
  README.md

  inputs/
    spreadsheets/
      jochen_field_notes_metadata_tables_with_geode_times.xlsx
      glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx

    surveys/
      T1_1m_refraction.csv
      T1_2m_refraction_hammer.csv
      T1_2m_betsy.csv
      T1_Streamer_MASW_main_transect.csv
      T1_N1_refraction.csv
      T1_N2_refraction.csv
      T2_Streamer_MASW_western_transect.csv
      T3_1m_refraction.csv
      T4_1m_refraction.csv

    receiver_components/
      T1_1m_refraction_receivers.csv
      T1_2m_refraction_receivers.csv
      T1_N1_geometry.csv
      T1_N2_geometry.csv
      T3_1m_refraction_receivers.csv
      T3_N4_geometry.csv
      T4_1m_refraction_receivers.csv
      streamer_24ch_5ft_relative.csv

    receivers/
      STATIONS_*.dat

    topography/
      T1_LiDAR_profile.csv
      T1_interfaces_0_300m_flat_subsurface.dat

    caves/
      T1_sunfish_regions_0p5m.inc
      T1_no_cave_regions.inc

    sources/
      source_families.csv

  templates/
    Par_file_T1_0_300.template
    SOURCE_ricker.template

  scripts/
    build_geometry_library_from_excel.py
    generate_specfem_runs.py

  generate_receivers_from_excel_metadata.sh
  generate_all_survey_run_suites.sh

  runs/
    T1/
    T2/
    T3/
    T4/
```

---

## 3. Step 1: build the geometry library

Run this first whenever the Excel metadata changes:

```bash
bash generate_receivers_from_excel_metadata.sh
```

This calls:

```bash
python scripts/build_geometry_library_from_excel.py \
  --jochen-xlsx inputs/spreadsheets/jochen_field_notes_metadata_tables_with_geode_times.xlsx \
  --nodal-xlsx inputs/spreadsheets/glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx \
  --topography inputs/topography/T1_LiDAR_profile.csv \
  --out-root inputs
```

It generates:

```text
inputs/surveys/
inputs/receiver_components/
inputs/receivers/
inputs/geometry_build_summary.json
```

### Current geometry conventions

| Survey | Shot list | Receiver components |
|---|---|---|
| T1 1-m refraction | `T1_1m_refraction.csv` | `T1_1m_refraction_receivers.csv` + `T1_N2_geometry.csv` |
| T1 2-m hammer | `T1_2m_refraction_hammer.csv` | `T1_2m_refraction_receivers.csv` |
| T1 Betsy | `T1_2m_betsy.csv` | `T1_2m_refraction_receivers.csv` |
| T1 streamer | `T1_Streamer_MASW_main_transect.csv` | moving streamer + `T1_N1_geometry.csv` |
| T1 N1 nodal-only | `T1_N1_refraction.csv` | `T1_N1_geometry.csv` |
| T1 N2 nodal-only | `T1_N2_refraction.csv` | `T1_N2_geometry.csv` |
| T2 streamer | `T2_Streamer_MASW_western_transect.csv` | moving streamer |
| T3 combined refraction/nodal | `T3_1m_refraction.csv` | `T3_1m_refraction_receivers.csv` + `T3_N4_geometry.csv` |
| T4 control refraction | `T4_1m_refraction.csv` | `T4_1m_refraction_receivers.csv` |

T3 has no separate nodal-only shot survey. The T3 shots were the shared 1-m refraction shots recorded by both the Geode line and T3_N4 nodes.

---

## 4. Step 2: generate SPECFEM run suites

After the geometry library exists, run:

```bash
bash generate_all_survey_run_suites.sh
```

This generates SPECFEM-ready run directories under:

```text
runs/
```

Each run folder contains:

```text
DATA/
  Par_file
  SOURCE
  STATIONS
  interfaces.dat

run_metadata.json
```

The run generator does not submit jobs.

---

## 5. Generated model suites

The current full wrapper generates:

| Line | Suite | Cave states | Notes |
|---|---|---|---|
| T1 | `T1_1m_refraction_plus_T1_N2` | with/no cave | 1-m Geode receivers plus dense N2 nodes |
| T1 | `T1_2m_refraction_hammer` | with/no cave | hammer shots only |
| T1 | `T1_2m_betsy` | with/no cave | single Betsy gun shot |
| T1 | `T1_Streamer_MASW_main_transect_plus_T1_N1` | with/no cave | moving streamer plus long N1 nodes |
| T1 | `T1_N1_refraction` | with/no cave | nodal-only long-offset shots |
| T1 | `T1_N2_refraction` | with/no cave | nodal-only dense-array shots |
| T2 | `T2_Streamer_MASW_western_transect` | no cave only | cave/topography not yet defined |
| T3 | `T3_1m_refraction_plus_T3_N4` | no cave only | combined Geode + N4 nodes |
| T4 | `T4_1m_refraction` | no cave only | control line |

For T2 and T3, with-cave models are intentionally skipped until line-specific cave polygons and topography/interface files are available.

---

## 6. Run-folder layout

Example:

```text
runs/T1/T1_2m_betsy/with_cave/shot_0002_x0047.00_betsy_force_vertical_f025/
  DATA/
    Par_file
    SOURCE
    STATIONS
    interfaces.dat
  run_metadata.json
```

For paired cave/no-cave modelling, the two folders should differ only in the cave material-region block.

---

## 7. Fixed versus moving receivers

### Fixed receiver surveys

For fixed spreads, receiver components are loaded from CSV files and written into `DATA/STATIONS`.

Examples:

```text
T1_1m_refraction_receivers.csv
T1_N2_geometry.csv
T3_N4_geometry.csv
```

Multiple receiver components can be combined on the fly. For example, T1 1-m refraction uses both Geode receivers and N2 nodes.

### Moving streamer surveys

For streamer surveys, the source and receivers move together.

The streamer component file is:

```text
inputs/receiver_components/streamer_24ch_5ft_relative.csv
```

It defines receiver offsets relative to the PEG source position:

```text
receiver_x = shot_x + offset_m
```

For streamer runs, `STATIONS` is regenerated for every shot.

---

## 8. Cave/no-cave modelling

For T1, paired models are generated:

```text
with_cave
no_cave
```

This enables direct differencing of synthetic shot gathers to isolate the cave-scattered wavefield.

Current T1 model:

```text
Horizontal extent: 0-300 m
Surface: LiDAR profile
Subsurface interfaces: flat
Only the upper slow layer has variable thickness
Cave: SunFISH-derived material-4 region file
```

Current cave-region files:

```text
inputs/caves/T1_sunfish_regions_0p5m.inc
inputs/caves/T1_no_cave_regions.inc
```

The cave should be represented using the SunFISH-derived polygon, not a generic centered rectangle.

---

## 9. Source families

Source definitions live in:

```text
inputs/sources/source_families.csv
templates/SOURCE_ricker.template
```

Current production placeholders:

| Source type | Source name |
|---|---|
| Hammer | `hammer_force_vertical_f050` |
| Betsy | `betsy_force_vertical_f025` |
| PEG | `peg_force_vertical_f050` |

These are working placeholders. Source calibration should be done as a separate exercise by comparing observed and synthetic shot gathers over a frequency/source-mechanism sweep.

For Betsy, possible calibration mechanisms include:

| Family | SPECFEM source type | Parameters |
|---|---:|---|
| Vertical force | `source_type = 1` | `anglesource = 180` |
| Isotropic explosion | `source_type = 2` | `Mxx = 1`, `Mzz = 1`, `Mxz = 0` |
| Vertically biased explosion | `source_type = 2` | `Mxx = 0.5`, `Mzz = 1`, `Mxz = 0` |
| Strong vertical bias | `source_type = 2` | `Mxx = 0.25`, `Mzz = 1`, `Mxz = 0` |

---

## 10. Quick checks after generation

Check the generated survey CSVs:

```bash
ls inputs/surveys
```

Check expected run counts:

```bash
find runs -mindepth 4 -maxdepth 4 -type d | wc -l
```

Spot-check a generated run:

```bash
find runs/T1/T1_2m_betsy -name Par_file -o -name SOURCE -o -name STATIONS | head

grep -n "use_existing_STATIONS" runs/T1/T1_2m_betsy/with_cave/*/DATA/Par_file
head runs/T1/T1_2m_betsy/with_cave/*/DATA/STATIONS
cat runs/T1/T1_2m_betsy/with_cave/*/DATA/SOURCE
```

Check that Betsy split is correct:

```bash
wc -l inputs/surveys/T1_2m_betsy.csv
wc -l inputs/surveys/T1_2m_refraction_hammer.csv
```

Expected:

```text
T1_2m_betsy.csv             header + 1 row
T1_2m_refraction_hammer.csv header + 41 rows
```

---

## 11. Known placeholders and future updates

Still needed later:

- T2 cave polygon and topography/interface files.
- T3 cave polygon and topography/interface files.
- T4 line-specific topography/interface file if needed.
- Source calibration for hammer, Betsy, and PEG.
- SLURM submission wrappers for CIRCE.
- Post-processing scripts to assemble synthetic shot gathers and difference with/no-cave runs.

---

## 12. Guiding principle

The modelling should mimic the actual field surveys as closely as practical:

- use actual shot positions;
- use actual occupied receiver positions;
- avoid interpolated or idealized acquisition geometries;
- use measured topography when available;
- use SunFISH-derived cave geometries where available;
- run paired cave/no-cave simulations for differencing.

This makes the synthetic data directly comparable with the field data trace-by-trace and keeps the workflow defensible for DEP delivery.
