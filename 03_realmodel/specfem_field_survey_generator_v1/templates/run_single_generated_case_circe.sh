#!/usr/bin/env bash
# Run one generated SPECFEM2D case on CIRCE.
#
# Intended location:
#   templates/run_single_generated_case_circe.sh
#
# Intended usage:
#   cd runs/T1/.../shot_XXXX_...
#   bash /path/to/specfem_field_survey_generator_v1/templates/run_single_generated_case_circe.sh
#
# Expected case directory contents:
#   DATA/Par_file
#   DATA/SOURCE
#   DATA/STATIONS
#   DATA/interfaces.dat
#
# Environment variables:
#   NPROC       MPI process count [default: 16]
#   BIN_DIR     SPECFEM2D bin directory [default: /shares/seismo_lab/specfem2d/bin]
#   CIRCE_FLAGS Extra mpirun flags [default: --mca btl self,vader,tcp]
#   FORCE_MESH  If 1, rerun xmeshfem2D even if mesh files appear to exist [default: 0]
#
# Outputs:
#   logs/meshfem.log
#   logs/specfem.log
#   logs/run_single_case.log

set -uo pipefail

NPROC="${NPROC:-16}"
BIN_DIR="${BIN_DIR:-/shares/seismo_lab/specfem2d/bin}"
CIRCE_FLAGS="${CIRCE_FLAGS:---mca btl self,vader,tcp}"
FORCE_MESH="${FORCE_MESH:-0}"

RUN_DIR="$(pwd)"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAIN_LOG="${LOG_DIR}/run_single_case.log"
MESHFEM_LOG="${LOG_DIR}/meshfem.log"
SPECFEM_LOG="${LOG_DIR}/specfem.log"

log_msg () {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

fail () {
  log_msg "ERROR: $*"
  exit 1
}

log_msg "============================================================"
log_msg "Starting generated SPECFEM2D case"
log_msg "RUN_DIR=${RUN_DIR}"
log_msg "NPROC=${NPROC}"
log_msg "BIN_DIR=${BIN_DIR}"
log_msg "CIRCE_FLAGS=${CIRCE_FLAGS}"
log_msg "FORCE_MESH=${FORCE_MESH}"
log_msg "============================================================"

# Required input files for generated cases.
[[ -f DATA/Par_file ]] || fail "Missing DATA/Par_file"
[[ -f DATA/SOURCE ]] || fail "Missing DATA/SOURCE"
[[ -f DATA/STATIONS ]] || fail "Missing DATA/STATIONS"

# interfaces.dat is expected for this workflow, but SPECFEM naming can vary by template.
# Treat it as required here because generated Par_file should point to DATA/interfaces.dat.
[[ -f DATA/interfaces.dat ]] || fail "Missing DATA/interfaces.dat"

# Required executables.
[[ -x "${BIN_DIR}/xmeshfem2D" ]] || fail "Executable not found: ${BIN_DIR}/xmeshfem2D"
[[ -x "${BIN_DIR}/xspecfem2D" ]] || fail "Executable not found: ${BIN_DIR}/xspecfem2D"

# Make sure SPECFEM output directories exist.
mkdir -p OUTPUT_FILES

# Decide whether mesh generation can be skipped.
# SPECFEM2D usually creates mesh files under OUTPUT_FILES/DATABASES_MPI or OUTPUT_FILES.
# The test below is intentionally conservative: if in doubt, remesh.
NEED_MESH=1
if [[ "${FORCE_MESH}" != "1" ]]; then
  if find OUTPUT_FILES -type f \( -name "*mesh*" -o -name "*Database*" -o -name "proc*_Database" \) | grep -q .; then
    NEED_MESH=0
  fi
fi

case_start_epoch="$(date +%s)"

if [[ "${NEED_MESH}" -eq 1 ]]; then
  log_msg "Running xmeshfem2D"
  mesh_start_epoch="$(date +%s)"

  "${BIN_DIR}/xmeshfem2D" > "${MESHFEM_LOG}" 2>&1
  mesh_status=$?

  mesh_end_epoch="$(date +%s)"
  log_msg "xmeshfem2D exit status: ${mesh_status}; elapsed seconds: $((mesh_end_epoch - mesh_start_epoch))"

  if [[ "${mesh_status}" -ne 0 ]]; then
    log_msg "xmeshfem2D failed. Last 40 lines of ${MESHFEM_LOG}:"
    tail -n 40 "${MESHFEM_LOG}" | tee -a "${MAIN_LOG}"
    exit "${mesh_status}"
  fi
else
  log_msg "Skipping xmeshfem2D because mesh files appear to exist. Set FORCE_MESH=1 to override."
fi

log_msg "Running xspecfem2D"
spec_start_epoch="$(date +%s)"

# shellcheck disable=SC2086
mpirun ${CIRCE_FLAGS} -np "${NPROC}" "${BIN_DIR}/xspecfem2D" > "${SPECFEM_LOG}" 2>&1
spec_status=$?

spec_end_epoch="$(date +%s)"
log_msg "xspecfem2D exit status: ${spec_status}; elapsed seconds: $((spec_end_epoch - spec_start_epoch))"

if [[ "${spec_status}" -ne 0 ]]; then
  log_msg "xspecfem2D failed. Last 40 lines of ${SPECFEM_LOG}:"
  tail -n 40 "${SPECFEM_LOG}" | tee -a "${MAIN_LOG}"
  exit "${spec_status}"
fi

case_end_epoch="$(date +%s)"
log_msg "Completed generated SPECFEM2D case successfully."
log_msg "Total elapsed seconds: $((case_end_epoch - case_start_epoch))"
log_msg "Logs:"
log_msg "  ${MAIN_LOG}"
log_msg "  ${MESHFEM_LOG}"
log_msg "  ${SPECFEM_LOG}"

exit 0
