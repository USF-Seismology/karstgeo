#!/bin/bash
set -e

#cd specfem_field_survey_generator_v1

echo "Building geometry library..."

python scripts/build_geometry_library_from_excel.py \
  --jochen-xlsx /Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/04_FieldData/jochen_field_notes_metadata_tables_with_geode_times.xlsx \
  --nodal-xlsx /Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/04_FieldData/glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx \
  --topography "/Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/05_Analysis/Elevation and DEMs/Profile_LIDAR_All_Transects.xlsx" \
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