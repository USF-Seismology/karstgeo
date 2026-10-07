#!/usr/bin/env bash
# Provisional four-rank SPECFEM2D validation job for USF CIRCE.
#
# Submit this file from an independent SPECFEM2D case directory containing:
#   DATA/Par_file
#   DATA/SOURCE
#   DATA/STATIONS
#   all model/interface files referenced by DATA/Par_file
#
# The NPROC value in DATA/Par_file must equal --ntasks below.
# This template uses the MPI-launched mesher that was verified on May 6, 2026.

#SBATCH --job-name=specfem2d-test
#SBATCH --account=normal
#SBATCH --partition=amd_2021
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load mpi/openmpi/3.1.6

cd "${SLURM_SUBMIT_DIR}"
mkdir -p OUTPUT_FILES logs

BIN_DIR=/shares/seismo_lab/specfem2d/bin
CIRCE_FLAGS="--mca btl self,vader,tcp"
NPROC="${SLURM_NTASKS}"

PAR_NPROC="$(awk -F= '
  /^[[:space:]]*NPROC[[:space:]]*=/ {
    value=$2
    sub(/#.*/, "", value)
    gsub(/[[:space:]]/, "", value)
    print value
    exit
  }
' DATA/Par_file)"

if [[ -z "${PAR_NPROC}" ]]; then
  echo "ERROR: Could not read NPROC from DATA/Par_file" >&2
  exit 2
fi

if [[ "${PAR_NPROC}" != "${NPROC}" ]]; then
  echo "ERROR: DATA/Par_file NPROC=${PAR_NPROC}; SLURM_NTASKS=${NPROC}" >&2
  exit 2
fi

if [[ ! -x "${BIN_DIR}/xmeshfem2D" || ! -x "${BIN_DIR}/xspecfem2D" ]]; then
  echo "ERROR: SPECFEM2D executables are unavailable under ${BIN_DIR}" >&2
  exit 3
fi

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Tasks: ${NPROC}"
module -t list 2>&1

mpirun ${CIRCE_FLAGS} -np "${NPROC}" \
  "${BIN_DIR}/xmeshfem2D" \
  > logs/meshfem.log 2>&1

mpirun ${CIRCE_FLAGS} -np "${NPROC}" \
  "${BIN_DIR}/xspecfem2D" \
  > logs/specfem.log 2>&1

echo "Completed: $(date --iso-8601=seconds)"

