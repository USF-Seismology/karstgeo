#!/usr/bin/env bash
# Run generated SPECFEM2D cases serially, with progress tracking and resume/skip logic.
#
# Intended location:
#   specfem_field_survey_generator_v1/scripts/run_generated_cases_serial.sh
#
# Intended usage from repository root:
#   bash scripts/run_generated_cases_serial.sh
#
# Useful options:
#   bash scripts/run_generated_cases_serial.sh --dry-run
#   bash scripts/run_generated_cases_serial.sh --run-root runs/T1/T1_2m_betsy
#   bash scripts/run_generated_cases_serial.sh --limit 10
#   bash scripts/run_generated_cases_serial.sh --force
#
# This script expects each case directory to contain:
#   DATA/Par_file
#   DATA/SOURCE
#   DATA/STATIONS
#
# It calls the single-case runner:
#   templates/run_single_generated_case_circe.sh
#
# Completion is detected by a marker file:
#   RUN_COMPLETE
#
# Failed cases get:
#   RUN_FAILED
#
# Progress logs are written under:
#   runs/progress/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="runs"
RUNNER="templates/run_single_generated_case_circe.sh"
DRY_RUN=0
FORCE=0
LIMIT=""
START_AT=""

usage() {
  cat <<EOF
Usage:
  bash scripts/run_generated_cases_serial.sh [options]

Options:
  --run-root DIR     Directory to crawl for generated cases [default: runs]
  --runner FILE      Single-case runner [default: templates/run_single_generated_case_circe.sh]
  --dry-run          Print cases that would run, but do not run them
  --force            Re-run cases even if RUN_COMPLETE exists
  --limit N          Run at most N pending cases
  --start-at PATH    Skip cases until this case directory is reached
  -h, --help         Show this help

Examples:
  bash scripts/run_generated_cases_serial.sh --dry-run
  bash scripts/run_generated_cases_serial.sh --run-root runs/T1/T1_2m_betsy
  bash scripts/run_generated_cases_serial.sh --limit 5
  bash scripts/run_generated_cases_serial.sh --force --run-root runs/T1/T1_2m_betsy
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --runner)
      RUNNER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --start-at)
      START_AT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "ERROR: run root not found: ${RUN_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: single-case runner not found: ${RUNNER}" >&2
  exit 1
fi

if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --limit must be an integer" >&2
  exit 1
fi

RUNNER_ABS="$(cd "$(dirname "${RUNNER}")" && pwd)/$(basename "${RUNNER}")"
PROGRESS_DIR="${RUN_ROOT}/progress"
mkdir -p "${PROGRESS_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_LIST="${PROGRESS_DIR}/case_list_${STAMP}.txt"
PENDING_LIST="${PROGRESS_DIR}/pending_cases_${STAMP}.txt"
SUCCESS_LOG="${PROGRESS_DIR}/success_${STAMP}.txt"
FAILED_LOG="${PROGRESS_DIR}/failed_${STAMP}.txt"
SKIPPED_LOG="${PROGRESS_DIR}/skipped_${STAMP}.txt"
SUMMARY_LOG="${PROGRESS_DIR}/summary_${STAMP}.txt"
LATEST_SUMMARY="${PROGRESS_DIR}/latest_summary.txt"

# Discover case directories from DATA/Par_file.
find "${RUN_ROOT}" -type f -path "*/DATA/Par_file" \
  | sed 's#/DATA/Par_file$##' \
  | sort > "${CASE_LIST}"

TOTAL_CASES="$(wc -l < "${CASE_LIST}" | tr -d ' ')"

if [[ "${TOTAL_CASES}" -eq 0 ]]; then
  echo "No generated SPECFEM cases found under ${RUN_ROOT}"
  exit 0
fi

# Build pending list, honoring RUN_COMPLETE unless --force.
: > "${PENDING_LIST}"
: > "${SKIPPED_LOG}"

STARTED=0
if [[ -z "${START_AT}" ]]; then
  STARTED=1
fi

while IFS= read -r case_dir; do
  if [[ "${STARTED}" -eq 0 ]]; then
    if [[ "${case_dir}" == "${START_AT}" || "${case_dir}" == *"${START_AT}"* ]]; then
      STARTED=1
    else
      echo "${case_dir}  SKIPPED_BEFORE_START_AT" >> "${SKIPPED_LOG}"
      continue
    fi
  fi

  if [[ "${FORCE}" -eq 0 && -f "${case_dir}/RUN_COMPLETE" ]]; then
    echo "${case_dir}  ALREADY_COMPLETE" >> "${SKIPPED_LOG}"
    continue
  fi

  echo "${case_dir}" >> "${PENDING_LIST}"
done < "${CASE_LIST}"

PENDING_CASES="$(wc -l < "${PENDING_LIST}" | tr -d ' ')"
ALREADY_SKIPPED="$(grep -c "ALREADY_COMPLETE" "${SKIPPED_LOG}" || true)"

{
  echo "Run discovery time: $(date)"
  echo "Repository root: ${REPO_ROOT}"
  echo "Run root: ${RUN_ROOT}"
  echo "Single-case runner: ${RUNNER_ABS}"
  echo "Total discovered cases: ${TOTAL_CASES}"
  echo "Already complete / skipped: ${ALREADY_SKIPPED}"
  echo "Pending cases: ${PENDING_CASES}"
  echo "Force rerun: ${FORCE}"
  echo "Dry run: ${DRY_RUN}"
  echo "Limit: ${LIMIT:-none}"
  echo
  echo "Case list: ${CASE_LIST}"
  echo "Pending list: ${PENDING_LIST}"
  echo "Success log: ${SUCCESS_LOG}"
  echo "Failed log: ${FAILED_LOG}"
  echo "Skipped log: ${SKIPPED_LOG}"
} | tee "${SUMMARY_LOG}" > "${LATEST_SUMMARY}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "Dry run: cases that would be run:"
  if [[ -n "${LIMIT}" ]]; then
    head -n "${LIMIT}" "${PENDING_LIST}"
  else
    cat "${PENDING_LIST}"
  fi
  echo
  echo "Dry run complete. Summary: ${SUMMARY_LOG}"
  exit 0
fi

: > "${SUCCESS_LOG}"
: > "${FAILED_LOG}"

RUN_COUNT=0
SUCCESS_COUNT=0
FAILED_COUNT=0

overall_start_epoch="$(date +%s)"

while IFS= read -r case_dir; do
  if [[ -n "${LIMIT}" && "${RUN_COUNT}" -ge "${LIMIT}" ]]; then
    break
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  remaining=$((PENDING_CASES - RUN_COUNT))

  echo
  echo "======================================================================"
  echo "Case ${RUN_COUNT}/${PENDING_CASES}: ${case_dir}"
  echo "Remaining after this case: ${remaining}"
  echo "Start time: $(date)"
  echo "======================================================================"

  case_start_epoch="$(date +%s)"

  # Remove stale failure marker before retrying.
  rm -f "${case_dir}/RUN_FAILED"

  (
    cd "${case_dir}"
    bash "${RUNNER_ABS}"
  )
  status=$?

  case_end_epoch="$(date +%s)"
  case_elapsed=$((case_end_epoch - case_start_epoch))

  if [[ "${status}" -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    {
      echo "case_dir=${case_dir}"
      echo "completed_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      echo "elapsed_seconds=${case_elapsed}"
      echo "runner=${RUNNER_ABS}"
    } > "${case_dir}/RUN_COMPLETE"
    rm -f "${case_dir}/RUN_FAILED"
    echo "${case_dir}  SUCCESS  elapsed_seconds=${case_elapsed}" >> "${SUCCESS_LOG}"
    echo "SUCCESS: ${case_dir} (${case_elapsed} s)"
  else
    FAILED_COUNT=$((FAILED_COUNT + 1))
    {
      echo "case_dir=${case_dir}"
      echo "failed_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      echo "elapsed_seconds=${case_elapsed}"
      echo "exit_status=${status}"
      echo "runner=${RUNNER_ABS}"
    } > "${case_dir}/RUN_FAILED"
    echo "${case_dir}  FAILED  status=${status}  elapsed_seconds=${case_elapsed}" >> "${FAILED_LOG}"
    echo "FAILED: ${case_dir} status=${status} (${case_elapsed} s)"
  fi

  elapsed_total=$((case_end_epoch - overall_start_epoch))

  {
    echo "Updated: $(date)"
    echo "Run root: ${RUN_ROOT}"
    echo "Total discovered cases: ${TOTAL_CASES}"
    echo "Pending at start: ${PENDING_CASES}"
    echo "Attempted this run: ${RUN_COUNT}"
    echo "Succeeded this run: ${SUCCESS_COUNT}"
    echo "Failed this run: ${FAILED_COUNT}"
    echo "Already skipped before run: ${ALREADY_SKIPPED}"
    echo "Elapsed seconds: ${elapsed_total}"
    if [[ "${RUN_COUNT}" -gt 0 ]]; then
      avg=$((elapsed_total / RUN_COUNT))
      echo "Average seconds per attempted case: ${avg}"
      echo "Estimated seconds for remaining pending cases: $((avg * (PENDING_CASES - RUN_COUNT)))"
    fi
    echo
    echo "Latest case: ${case_dir}"
    echo "Latest case status: $([[ "${status}" -eq 0 ]] && echo SUCCESS || echo FAILED)"
    echo
    echo "Success log: ${SUCCESS_LOG}"
    echo "Failed log: ${FAILED_LOG}"
    echo "Skipped log: ${SKIPPED_LOG}"
  } | tee "${LATEST_SUMMARY}" > /dev/null

done < "${PENDING_LIST}"

overall_end_epoch="$(date +%s)"
overall_elapsed=$((overall_end_epoch - overall_start_epoch))

{
  echo "Final summary time: $(date)"
  echo "Repository root: ${REPO_ROOT}"
  echo "Run root: ${RUN_ROOT}"
  echo "Total discovered cases: ${TOTAL_CASES}"
  echo "Pending at start: ${PENDING_CASES}"
  echo "Attempted this run: ${RUN_COUNT}"
  echo "Succeeded this run: ${SUCCESS_COUNT}"
  echo "Failed this run: ${FAILED_COUNT}"
  echo "Already skipped before run: ${ALREADY_SKIPPED}"
  echo "Elapsed seconds: ${overall_elapsed}"
  echo
  echo "Case list: ${CASE_LIST}"
  echo "Pending list: ${PENDING_LIST}"
  echo "Success log: ${SUCCESS_LOG}"
  echo "Failed log: ${FAILED_LOG}"
  echo "Skipped log: ${SKIPPED_LOG}"
} | tee "${SUMMARY_LOG}" | tee "${LATEST_SUMMARY}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  echo
  echo "Some cases failed. See: ${FAILED_LOG}"
  exit 2
fi

echo
echo "All attempted cases completed successfully."
