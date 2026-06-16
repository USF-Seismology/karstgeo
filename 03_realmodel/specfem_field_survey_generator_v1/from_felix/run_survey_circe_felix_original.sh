#!/bin/bash
# =============================================================================
# run_survey_circe.sh — Adapted for USF CIRCE HPC
# =============================================================================

set -euo pipefail
# ── Configuration ─────────────────────────────────────────────────────────────

NPROC=16                 # Match your CIRCE MPI cores
XS_START=82.5            # First shot position (m)
XS_STEP=2.0              # Shot spacing (m)
NSHOTS=41                # Total number of shots (ends at 162.5m)
SHOT_START=1             # Set > 1 to resume from a specific shot
RUN_MESHER=true          # Set false to skip mesher (if resuming)

# MODIFICATION 1: Define the path to Glenn's verified binaries
BIN_DIR="/shares/seismo_lab/specfem2d/bin"
# MODIFICATION 2: Define the "Glenn Flags" for CIRCE network stability
CIRCE_FLAGS="--mca btl self,vader,tcp"

SURVEY_DIR="SURVEY_OUTPUT"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -f "DATA/Par_file" ]; then
    echo "ERROR: DATA/Par_file not found."
    exit 1
fi

mkdir -p "$SURVEY_DIR"

# ── Step 1: Run mesher once ───────────────────────────────────────────────────

if [ "$RUN_MESHER" = true ]; then
    echo "============================================================"
    echo " Building mesh on CIRCE ..."
    echo "============================================================"
    sed "s/XS_PLACEHOLDER/$XS_START/" DATA/SOURCE_template > DATA/SOURCE
    
    # MODIFICATION 3: Run mesher with MPI and CIRCE flags
    "$BIN_DIR/xmeshfem2D"
    
    echo " Mesh complete."
    echo ""
fi

# ── Step 2: Loop over shots ───────────────────────────────────────────────────
SURVEY_START_TIME=$SECONDS

for SHOT in $(seq "$SHOT_START" "$NSHOTS"); do

    XS=$(echo "scale=1; $XS_START + ($SHOT - 1) * $XS_STEP" | bc)
    XS_TAG=$(printf '%07.1f' "$XS" | tr '.' 'p')
    SHOT_DIR="$SURVEY_DIR/shot_$(printf '%03d' $SHOT)_xs${XS_TAG}"

    printf "Shot %03d/%d  xs=%6.1f m  ->  %s ... " "$SHOT" "$NSHOTS" "$XS" "$SHOT_DIR"

    sed "s/XS_PLACEHOLDER/$XS/" DATA/SOURCE_template > DATA/SOURCE

    # MODIFICATION 4: Run solver with Glenn's verified binary and CIRCE flags
    mpirun $CIRCE_FLAGS -np "$NPROC" "$BIN_DIR/xspecfem2D" > /dev/null 2>&1

    # Archive seismograms
    mkdir -p "$SHOT_DIR"
    mv OUTPUT_FILES/*.su "$SHOT_DIR/" 2>/dev/null || echo "  WARNING: no .su files"

    # Progress estimate
    ELAPSED=$(( SECONDS - SURVEY_START_TIME ))
    SHOTS_DONE=$(( SHOT - SHOT_START + 1 ))
    AVG=$(( ELAPSED / SHOTS_DONE ))
    REMAINING=$(( AVG * (NSHOTS - SHOT) ))
    printf "done  [%ds elapsed, ~%dh%02dm remaining]\n" "$ELAPSED" "$(( REMAINING/3600 ))" "$(( (REMAINING%3600)/60 ))"
done

echo "Survey complete."
