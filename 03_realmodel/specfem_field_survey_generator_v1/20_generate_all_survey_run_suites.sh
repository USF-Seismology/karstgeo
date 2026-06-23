#!/usr/bin/env bash
# Generate SPECFEM2D model-run suites using the generated geometry library.
#
# Assumes:
#   1. scripts/build_geometry_library_from_excel.py has already been run.
#   2. inputs/surveys/ and inputs/receiver_components/ exist.
#   3. scripts/generate_specfem_runs.py is the component-based version.
#
# This script only generates SPECFEM run directories. It does not submit jobs.

set -euo pipefail

# This script is intended to live in the repository root:
#   specfem_field_survey_generator_v1/generate_all_survey_run_suites.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

echo "PWD=$(pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "Generating SPECFEM run suites..."

GEN="python scripts/generate_specfem_runs.py"

# Chosen placeholder source models.
# Source calibration should be done separately.
HAMMER_SOURCE="hammer_force_vertical_f050"
BETSY_SOURCE="betsy_force_vertical_f025"
PEG_SOURCE="peg_force_vertical_f050"

# T1 physical model files.
TOPO_WORKBOOK="inputs/topography/Profile - All Transect .xlsx"
T1_INTERFACES="inputs/topography/T1_interfaces_0_300m_flat_subsurface.dat"
T1_CAVE="inputs/caves/T1_sunfish_regions_0p5m.inc"
T1_NO_CAVE="inputs/caves/T1_no_cave_regions.inc"

# Placeholder model files for lines without finalized topography/cave products.
FLAT_INTERFACES="${T1_INTERFACES}"
NO_CAVE_REGIONS="${T1_NO_CAVE}"

require_file () {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: required file not found: ${f}" >&2
    exit 1
  fi
}

optional_file_exists () {
  [[ -f "$1" ]]
}

generate_fixed_pair () {
  local line="$1"
  local survey_csv="$2"
  local survey_name="$3"
  local source_name="$4"
  local refraction_geom="${5:-}"
  local nodal_geom="${6:-}"
  local interfaces_file="$7"
  local cave_file="$8"
  local no_cave_file="$9"
  local topography_file="${10:-}"

  require_file "${survey_csv}"
  require_file "${interfaces_file}"
  require_file "${no_cave_file}"

  local recv_args=()
  if [[ -n "${refraction_geom}" ]]; then
    require_file "${refraction_geom}"
    recv_args+=(--refraction-geometry "${refraction_geom}")
  fi

  if [[ -n "${nodal_geom}" ]]; then
    require_file "${nodal_geom}"
    recv_args+=(--nodal-geometry "${nodal_geom}")
  fi

  local topo_args=()
  if [[ -n "${topography_file}" ]]; then
    require_file "${topography_file}"
    topo_args+=(--topography "${topography_file}")
  fi

  if [[ -n "${cave_file}" && -f "${cave_file}" ]]; then
    echo "Generating ${line}/${survey_name}/with_cave"
    ${GEN} \
      --survey-csv "${survey_csv}" \
      ${recv_args[@]+"${recv_args[@]}"} \
      ${topo_args[@]+"${topo_args[@]}"} \
      --survey-name "${survey_name}" \
      --line "${line}" \
      --cave-state with_cave \
      --interfaces-file "${interfaces_file}" \
      --cave-regions-file "${cave_file}" \
      --no-cave-regions-file "${no_cave_file}" \
      --single-source-name "${source_name}"
  else
    echo "Skipping ${line}/${survey_name}/with_cave: no cave file supplied/found"
  fi

  echo "Generating ${line}/${survey_name}/no_cave"
  ${GEN} \
    --survey-csv "${survey_csv}" \
    ${recv_args[@]+"${recv_args[@]}"} \
    ${topo_args[@]+"${topo_args[@]}"} \
    --survey-name "${survey_name}" \
    --line "${line}" \
    --cave-state no_cave \
    --interfaces-file "${interfaces_file}" \
    --cave-regions-file "${cave_file:-${no_cave_file}}" \
    --no-cave-regions-file "${no_cave_file}" \
    --single-source-name "${source_name}"
}

generate_streamer_pair () {
  local line="$1"
  local survey_csv="$2"
  local survey_name="$3"
  local source_name="$4"
  local streamer_geom="$5"
  local nodal_geom="${6:-}"
  local interfaces_file="$7"
  local cave_file="$8"
  local no_cave_file="$9"
  local topography_file="${10:-}"

  require_file "${survey_csv}"
  require_file "${streamer_geom}"
  require_file "${interfaces_file}"
  require_file "${no_cave_file}"

  local recv_args=(--streamer-geometry "${streamer_geom}")
  if [[ -n "${nodal_geom}" ]]; then
    require_file "${nodal_geom}"
    recv_args+=(--nodal-geometry "${nodal_geom}")
  fi

  local topo_args=()
  if [[ -n "${topography_file}" ]]; then
    require_file "${topography_file}"
    topo_args+=(--topography "${topography_file}")
  fi

  if [[ -n "${cave_file}" && -f "${cave_file}" ]]; then
    echo "Generating ${line}/${survey_name}/with_cave"
    ${GEN} \
      --survey-csv "${survey_csv}" \
      ${recv_args[@]+"${recv_args[@]}"} \
      ${topo_args[@]+"${topo_args[@]}"} \
      --survey-name "${survey_name}" \
      --line "${line}" \
      --cave-state with_cave \
      --interfaces-file "${interfaces_file}" \
      --cave-regions-file "${cave_file}" \
      --no-cave-regions-file "${no_cave_file}" \
      --single-source-name "${source_name}"
  else
    echo "Skipping ${line}/${survey_name}/with_cave: no cave file supplied/found"
  fi

  echo "Generating ${line}/${survey_name}/no_cave"
  ${GEN} \
    --survey-csv "${survey_csv}" \
    ${recv_args[@]+"${recv_args[@]}"} \
    ${topo_args[@]+"${topo_args[@]}"} \
    --survey-name "${survey_name}" \
    --line "${line}" \
    --cave-state no_cave \
    --interfaces-file "${interfaces_file}" \
    --cave-regions-file "${cave_file:-${no_cave_file}}" \
    --no-cave-regions-file "${no_cave_file}" \
    --single-source-name "${source_name}"
}

# ------------------------------------------------------------------------------
# T1: cave line, current best model available
# ------------------------------------------------------------------------------

generate_fixed_pair \
  T1 \
  inputs/surveys/T1_1m_refraction.csv \
  T1_1m_refraction_plus_T1_N2 \
  "${HAMMER_SOURCE}" \
  inputs/receiver_components/T1_1m_refraction_receivers.csv \
  inputs/receiver_components/T1_N2_geometry.csv \
  "${T1_INTERFACES}" \
  "${T1_CAVE}" \
  "${T1_NO_CAVE}" \
  "${TOPO_WORKBOOK}"

generate_fixed_pair \
  T1 \
  inputs/surveys/T1_2m_refraction_hammer.csv \
  T1_2m_refraction_hammer \
  "${HAMMER_SOURCE}" \
  inputs/receiver_components/T1_2m_refraction_receivers.csv \
  "" \
  "${T1_INTERFACES}" \
  "${T1_CAVE}" \
  "${T1_NO_CAVE}" \
  "${TOPO_WORKBOOK}"

generate_fixed_pair \
  T1 \
  inputs/surveys/T1_2m_betsy.csv \
  T1_2m_betsy \
  "${BETSY_SOURCE}" \
  inputs/receiver_components/T1_2m_refraction_receivers.csv \
  "" \
  "${T1_INTERFACES}" \
  "${T1_CAVE}" \
  "${T1_NO_CAVE}" \
  "${TOPO_WORKBOOK}"

if optional_file_exists inputs/surveys/T1_Streamer_MASW_main_transect.csv; then
  generate_streamer_pair \
    T1 \
    inputs/surveys/T1_Streamer_MASW_main_transect.csv \
    T1_Streamer_MASW_main_transect_plus_T1_N1 \
    "${PEG_SOURCE}" \
    inputs/receiver_components/streamer_24ch_5ft_relative.csv \
    inputs/receiver_components/T1_N1_geometry.csv \
    "${T1_INTERFACES}" \
    "${T1_CAVE}" \
    "${T1_NO_CAVE}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T1 streamer: inputs/surveys/T1_Streamer_MASW_main_transect.csv not found"
fi

if optional_file_exists inputs/surveys/T1_N1_refraction.csv; then
  generate_fixed_pair \
    T1 \
    inputs/surveys/T1_N1_refraction.csv \
    T1_N1_refraction \
    "${HAMMER_SOURCE}" \
    "" \
    inputs/receiver_components/T1_N1_geometry.csv \
    "${T1_INTERFACES}" \
    "${T1_CAVE}" \
    "${T1_NO_CAVE}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T1_N1_refraction: inputs/surveys/T1_N1_refraction.csv not found"
fi

if optional_file_exists inputs/surveys/T1_N2_refraction.csv; then
  generate_fixed_pair \
    T1 \
    inputs/surveys/T1_N2_refraction.csv \
    T1_N2_refraction \
    "${HAMMER_SOURCE}" \
    "" \
    inputs/receiver_components/T1_N2_geometry.csv \
    "${T1_INTERFACES}" \
    "${T1_CAVE}" \
    "${T1_NO_CAVE}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T1_N2_refraction: inputs/surveys/T1_N2_refraction.csv not found"
fi

# ------------------------------------------------------------------------------
# T2: streamer line, currently no finalized T2 cave/topography model.
# ------------------------------------------------------------------------------

if optional_file_exists inputs/surveys/T2_Streamer_MASW_western_transect.csv; then
  generate_streamer_pair \
    T2 \
    inputs/surveys/T2_Streamer_MASW_western_transect.csv \
    T2_Streamer_MASW_western_transect \
    "${PEG_SOURCE}" \
    inputs/receiver_components/streamer_24ch_5ft_relative.csv \
    "" \
    "${FLAT_INTERFACES}" \
    "" \
    "${NO_CAVE_REGIONS}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T2 streamer: inputs/surveys/T2_Streamer_MASW_western_transect.csv not found"
fi

# ------------------------------------------------------------------------------
# T3: combined 1-m refraction + T3_N4 nodal receivers.
# The only T3 shots were the shared 1-m refraction shots, recorded by both
# Geode receivers and the T3_N4 nodal array. No T3 cave/topography model yet,
# so generate no-cave only.
# ------------------------------------------------------------------------------

if optional_file_exists inputs/surveys/T3_1m_refraction.csv; then
  generate_fixed_pair \
    T3 \
    inputs/surveys/T3_1m_refraction.csv \
    T3_1m_refraction_plus_T3_N4 \
    "${HAMMER_SOURCE}" \
    inputs/receiver_components/T3_1m_refraction_receivers.csv \
    inputs/receiver_components/T3_N4_geometry.csv \
    "${FLAT_INTERFACES}" \
    "" \
    "${NO_CAVE_REGIONS}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T3_1m_refraction_plus_T3_N4: inputs/surveys/T3_1m_refraction.csv not found"
fi

# ------------------------------------------------------------------------------
# T4: control refraction line, no cave.
# ------------------------------------------------------------------------------

if optional_file_exists inputs/surveys/T4_1m_refraction.csv; then
  generate_fixed_pair \
    T4 \
    inputs/surveys/T4_1m_refraction.csv \
    T4_1m_refraction \
    "${HAMMER_SOURCE}" \
    inputs/receiver_components/T4_1m_refraction_receivers.csv \
    "" \
    "${FLAT_INTERFACES}" \
    "" \
    "${NO_CAVE_REGIONS}" \
    "${TOPO_WORKBOOK}"
else
  echo "Skipping T4_1m_refraction: inputs/surveys/T4_1m_refraction.csv not found"
fi

echo "Done generating model run suites under runs/"
