#!/bin/bash
set -e

#cd specfem_field_survey_generator_v1

echo "Building geometry library..."

python scripts/build_geometry_library_from_excel.py \
  --jochen-xlsx inputs/spreadsheets/jochen_field_notes_metadata_tables_with_geode_times.xlsx \
  --nodal-xlsx inputs/spreadsheets/glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx \
  --topography "inputs/topography/Profile - All Transect .xlsx" \
  --out-root inputs

echo
echo "Generated surveys:"
ls -1 inputs/surveys

echo
echo "Generated receiver components:"
ls -1 inputs/receiver_components

echo
echo "Generated STATIONS files:"
ls -1 inputs/receivers

echo
echo "Done."