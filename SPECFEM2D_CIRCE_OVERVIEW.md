# SPECFEM2D on USF CIRCE

## Purpose

This document records the known location, provenance, compilation environment, runtime requirements, and survey workflow for the shared SPECFEM2D installation on the USF CIRCE research cluster.

It is based on:

- the working CIRCE installation inspected in October 2026;
- compilation notes emailed on May 6, 2026;
- the successful CIRCE example run described in that email;
- Felix's original CIRCE survey script; and
- the newer generated-case runner and field-survey model generator.

Items that have not yet been verified are identified explicitly.

---

## 1. Access and installation location

Log in to CIRCE with:

```bash
ssh USERNAME@circe.rc.usf.edu
```

The shared SPECFEM2D installation is located at:

```text
/shares/seismo_lab/specfem2d
```

For members of the `seismo_lab` group, it is also available as:

```text
~/seismo_lab/specfem2d
```

The latter resolves to the shared location:

```bash
cd ~/seismo_lab/specfem2d
readlink -f .
```

Expected result:

```text
/shares/seismo_lab/specfem2d
```

The installation is a complete SPECFEM2D source checkout, not just a collection of executables.

### Principal directories

```text
/shares/seismo_lab/specfem2d/
├── bin
├── DATA
├── doc
├── EXAMPLES
├── external_libs
├── m4
├── obj
├── OUTPUT_FILES
├── setup
├── src
├── tests
└── utils
```

### Executables

The main executables are:

```text
/shares/seismo_lab/specfem2d/bin/xmeshfem2D
/shares/seismo_lab/specfem2d/bin/xspecfem2D
```

Additional compiled utilities include:

```text
xadj_seismogram
xcheck_quality_external_mesh
xconvolve_source_timefunction
```

The executables are owned by `thompsong:seismo_lab` and were compiled on May 6, 2026. Their permissions are `0750`, allowing execution by the owner and members of the `seismo_lab` group.

| Executable | Size | Modification time (EDT) |
|---|---:|---|
| `xmeshfem2D` | 800,144 bytes | 2026-05-06 19:07:50 |
| `xspecfem2D` | 4,995,480 bytes | 2026-05-06 19:07:55 |

---

## 2. Source provenance

The source repository is:

```text
https://github.com/SPECFEM/specfem2d.git
```

The installed checkout was observed at commit:

```text
abf3196e3192af3343f2b402c82f34205577e672
```

Commit information:

```text
Date:    2025-12-17 19:46:34 +0100
Subject: fixes null pointer in JPEG library (CVE GHSA-xggf-8r3g-7cvj)
Branch:  master
```

The installed SPECFEM2D version is:

```text
8.1.0
```

The checkout describes itself as:

```text
v8.1.0-4-gabf3196-dirty
```

This means that it is four commits beyond tag `v8.1.0` and contains local changes.

At the time of inspection, the shared checkout was not clean:

```text
Modified:   DATA/Par_file
Untracked:  DATA/STATIONS
Generated:  setup/config.fh
            setup/constants_tomography.h
            setup/version.fh
            test_mpi
            test_mpi.f90
Deleted:    bin/.keep
            obj/.keep
```

Some of these changes are normal consequences of configuration, compilation, and running a model. Nevertheless, do **not** use `git clean`, `git reset --hard`, or otherwise reset this shared checkout without first preserving all runtime inputs and confirming that no one else is using it.

---

## 3. Verified compilation environment

The principal compilation problem on CIRCE was an MPI/compiler mismatch. The following environment successfully compiled SPECFEM2D:

```bash
module purge
module load mpi/openmpi/3.1.6
```

Verification performed at the time gave:

```text
/apps/openmpi/3.1.6/bin/mpifort
GNU Fortran (GCC) 4.8.5
```

The retained `config.log` confirms that configuration was performed on `itn1.rc.usf.edu`, using SPECFEM2D 8.1.0 and GNU Autoconf 2.71. The build platform was `x86_64-unknown-linux-gnu`, and the underlying Fortran compiler was:

```text
GNU Fortran 4.8.5 20150623 (Red Hat 4.8.5-28)
```

The installed `xspecfem2D` is dynamically linked against:

```text
/apps/openmpi/3.1.6/lib/libmpi_usempi.so.40
/apps/openmpi/3.1.6/lib/libmpi_mpifh.so.40
/apps/openmpi/3.1.6/lib/libmpi.so.40
/lib64/libgfortran.so.3
```

This independently confirms that the working executable uses OpenMPI 3.1.6 and the GNU Fortran runtime.

### Exact compilation configuration used on May 6, 2026

From the SPECFEM2D source directory:

```bash
cd /shares/seismo_lab/specfem2d

./configure FC=mpifort CC=mpicc MPIFC=mpifort MPICC=mpicc --with-mpi
make clean
make all
```

This exact invocation is preserved independently in both `config.log` and `config.status`:

```text
FC=mpifort CC=mpicc MPIFC=mpifort MPICC=mpicc --with-mpi
```

The May 6 email abbreviated the configuration as `CC=gcc` and did not show `MPICC` or `--with-mpi`. The retained build files are authoritative: the existing executables were configured with the MPI C wrapper, `mpicc`, and explicit MPI support.

The commands above document how the existing binaries were produced. They should **not** be rerun casually in the shared working installation.

### OpenMPI wrapper configuration

The OpenMPI module adds the following directories and variables:

```text
PATH:             /apps/openmpi/3.1.6/bin
LD_LIBRARY_PATH:  /apps/openmpi/3.1.6/lib
MANPATH:          /apps/openmpi/3.1.6/share/man
ORTE_PATH:        /apps/openmpi/3.1.6
SLURM_OVERLAP:    1
```

`mpifort` invokes `gfortran` with:

```text
Compile: -I/apps/openmpi/3.1.6/include -pthread -I/apps/openmpi/3.1.6/lib
Link:    -pthread -I/apps/openmpi/3.1.6/lib -Wl,-rpath \
         -Wl,/apps/openmpi/3.1.6/lib -Wl,--enable-new-dtags \
         -L/apps/openmpi/3.1.6/lib -lmpi_usempi -lmpi_mpifh -lmpi
```

The embedded runtime path helps the executables locate the matching OpenMPI libraries, but the OpenMPI 3.1.6 module should still be loaded before running them.

### Compiled features and dependencies

The generated `setup/config.h` explicitly reports:

```c
#define HAVE_SCOTCH 1
```

Thus, Scotch support was detected during configuration. A search for the selected MPI, CUDA, OpenCL, ADIOS, VTK, and JPEG feature names returned no other matching definitions in `setup/config.h`; MPI support is nevertheless established by the configure invocation and the linked MPI libraries.

Both primary executables resolve all of the libraries reported by `ldd`. Their shared dependencies include:

- OpenMPI 3.1.6 Fortran and core MPI libraries;
- GNU Fortran, GCC, and Quadmath runtimes;
- POSIX threading, math, C, dynamic-loading, real-time, and utility libraries;
- OpenMPI runtime and portability layers;
- NUMA support; and
- zlib.

No missing shared library was reported during the October 7, 2026 inspection.

### Executable fingerprints

The compiled executables are dynamically linked, unstripped, 64-bit x86-64 ELF files. Their recorded identities are:

| Executable | SHA-256 | ELF Build ID |
|---|---|---|
| `xmeshfem2D` | `161dc5407c1e9b07ee7e3a96b261430f041517b6ba490ecc96e8be14d5770024` | `8697f37836ce0f6bed610e6eb16e1d9dcd1d824d` |
| `xspecfem2D` | `13838a93527c16990ccdab50c0183fd6568934582f399fe2667a6c55860618b2` | `b2314af760d4cd966b5658e74593ad1bdc76c2a6` |

These fingerprints should be stored with production-run metadata. A changed checksum means that the executable is not byte-for-byte identical to the verified May 6 build.

### Recommended recompilation practice

If recompilation becomes necessary:

1. Leave `/shares/seismo_lab/specfem2d` unchanged as the known working installation.
2. Create a separate checkout or build directory.
3. Record the Git commit, loaded modules, compiler versions, complete `configure` command, and build log.
4. Run the standard example before attempting the cave models.
5. Compare the new executable dependencies and checksums with the established installation.
6. Replace the shared binaries only after the new build has been independently verified.

---

## 4. Runtime environment on CIRCE

Load the same MPI implementation used during compilation:

```bash
module purge
module load mpi/openmpi/3.1.6
```

Check the runtime before starting a simulation:

```bash
which mpirun
mpirun --version
which mpifort
mpifort --version
```

### CIRCE OpenMPI workaround

OpenMPI may attempt to use InfiniBand verbs on CIRCE and report an error similar to:

```text
Failed to register memory region (MR)
```

The verified workaround is to restrict OpenMPI to the self, shared-memory, and TCP transports:

```bash
--mca btl self,vader,tcp
```

The resulting solver command is:

```bash
mpirun --mca btl self,vader,tcp -np NPROC \
  /shares/seismo_lab/specfem2d/bin/xspecfem2D
```

`NPROC` on the command line must agree with `NPROC` in `DATA/Par_file`.

An occasional warning was reportedly still printed with this workaround, but the May 6 test completed successfully.

---

## 5. Verified example

The following bundled example was successfully run on four MPI ranks:

```text
EXAMPLES/simple_topography_and_also_a_simple_fluid_layer
```

It exercised:

- MPI partitioning;
- acoustic-elastic coupling;
- perfectly matched layer boundaries;
- wavefield snapshots;
- seismogram output;
- JPEG movie frames; and
- VTK output.

The May 6 validation used:

```bash
mpirun --mca btl self,vader,tcp -np 4 ./bin/xmeshfem2D
mpirun --mca btl self,vader,tcp -np 4 ./bin/xspecfem2D
```

---

## 6. Required model inputs

A normal case directory for the current cave-survey workflow contains:

```text
CASE_DIRECTORY/
├── DATA/
│   ├── Par_file
│   ├── SOURCE
│   ├── STATIONS
│   └── interfaces.dat
├── OUTPUT_FILES/
└── run_metadata.json       # generated workflow only
```

The precise auxiliary files depend on the options and filenames referenced by `DATA/Par_file`. Always inspect `Par_file` rather than assuming that `interfaces.dat` is the only external model file.

### Roles of the principal files

- `Par_file`: numerical settings, simulation type, processor count, model and mesh settings, output controls, and filenames.
- `SOURCE`: source location, mechanism, time function, dominant frequency, and timing.
- `STATIONS`: receiver names, networks, coordinates, and elevations/depths.
- `interfaces.dat`: interface geometry used by the current layered models.
- material and region definitions: assign seismic properties to mesh regions; in the cave/no-cave workflow these definitions distinguish the paired models.

Before running, confirm that all paths referenced by `Par_file` are valid relative to the case directory.

### Current shared `DATA/Par_file`

As inspected on October 7, 2026, the shared source-tree `DATA/Par_file` is still configured for the bundled curved-interface test rather than a cave-production model:

```text
title                  = Test of SPECFEM2D with curved interfaces
SIMULATION_TYPE        = 1
SAVE_FORWARD           = .false.
NPROC                  = 4
MODEL                  = default
SAVE_MODEL             = default
use_existing_STATIONS  = .false.
interfacesfile         = ../EXAMPLES/simple_topography_and_also_a_simple_fluid_layer/DATA/interfaces_simple_topo_curved.dat
BROADCAST_SAME_MESH_AND_MODEL = .true.
```

The only tracked change in this file is:

```diff
-NPROC = 1
+NPROC = 4
```

This confirms that the shared root retains the successful four-rank validation configuration. It should not be treated as the authoritative Mod15 or field-survey input directory.

---

## 7. Meshing and solving

### Initial test procedure

The May 6 example was validated with both programs launched through MPI:

```bash
mpirun --mca btl self,vader,tcp -np 4 \
  /shares/seismo_lab/specfem2d/bin/xmeshfem2D

mpirun --mca btl self,vader,tcp -np 4 \
  /shares/seismo_lab/specfem2d/bin/xspecfem2D
```

### Required MPI behavior

The bundled example's `run_this_example.sh` confirms the intended behavior:

- if `NPROC = 1`, launch both programs directly; and
- if `NPROC > 1`, launch both programs through `mpirun -np NPROC`.

On CIRCE, the parallel form is therefore:

```bash
mpirun --mca btl self,vader,tcp -np "${NPROC}" \
  /shares/seismo_lab/specfem2d/bin/xmeshfem2D

mpirun --mca btl self,vader,tcp -np "${NPROC}" \
  /shares/seismo_lab/specfem2d/bin/xspecfem2D
```

The process count must equal `NPROC` in `DATA/Par_file` for both commands.

### Survey-runner correction required

Felix's CIRCE survey script and the newer generated-case runner launch the mesher directly and the solver with MPI:

```bash
/shares/seismo_lab/specfem2d/bin/xmeshfem2D

mpirun --mca btl self,vader,tcp -np 16 \
  /shares/seismo_lab/specfem2d/bin/xspecfem2D
```

This is inconsistent with the bundled parallel example and the successful May 6 validation. Felix's script even says that the mesher is run with MPI, while its actual command launches it directly.

Before using either runner with `NPROC > 1`, change its mesher command to:

```bash
mpirun ${CIRCE_FLAGS} -np "${NPROC}" "${BIN_DIR}/xmeshfem2D"
```

Preserve the mesher and solver commands and logs with every validation run.

### When remeshing is required

Remesh whenever a change affects the computational mesh or its partitioning, including changes to:

- the computational domain;
- topography or interfaces;
- region geometry, including cave geometry;
- element spacing or mesh resolution;
- boundary conditions that are encoded during meshing; or
- `NPROC`, if it changes the database partitioning.

Changing only the source position normally allows the existing mesh and databases to be reused. Confirm this with the current model before launching a complete survey.

---

## 8. Survey workflow

Sarah's Mod15 workflow and Felix's adapted CIRCE script use the following pattern:

1. Prepare and verify the model inputs.
2. Generate the mesh once.
3. Update `DATA/SOURCE` for each shot.
4. Run `xspecfem2D` for that shot.
5. Copy or move the seismograms into a shot-specific directory.
6. Continue to the next source position.

Felix's original CIRCE script used:

```text
NPROC:       16
First shot:  82.5 m
Spacing:     2.0 m
Shots:       41
Last shot:   162.5 m
```

It archived SU seismograms under directories resembling:

```text
SURVEY_OUTPUT/shot_001_xs0082p5/
```

The script can resume at a selected shot and can skip meshing when continuing a previously meshed survey.

### Important output rule

SPECFEM2D may overwrite files in `OUTPUT_FILES` during the next shot. Archive all required output immediately after every successful solver run.

Do not limit archiving to `*.su` until it has been confirmed that no other products are needed. Preserve logs, metadata, source files, and model identifiers alongside the seismograms.

---

## 9. Generated field-survey cases

The newer field-survey generator creates independent run directories containing:

```text
DATA/Par_file
DATA/SOURCE
DATA/STATIONS
DATA/interfaces.dat
run_metadata.json
```

The single-case CIRCE runner uses these defaults:

```text
NPROC=16
BIN_DIR=/shares/seismo_lab/specfem2d/bin
CIRCE_FLAGS="--mca btl self,vader,tcp"
FORCE_MESH=0
```

From a generated case directory, it is invoked as:

```bash
bash /path/to/specfem_field_survey_generator_v1/templates/run_single_generated_case_circe.sh
```

It writes:

```text
logs/meshfem.log
logs/specfem.log
logs/run_single_case.log
```

The serial multi-case controller supports discovery, dry runs, limits, restart markers, and success/failure logs. Its internal path handling currently expects the file to be located at:

```text
scripts/run_generated_cases_serial.sh
```

However, the current local copy is stored at the repository root as:

```text
40_run_generated_cases_serial.sh
```

That location mismatch must be corrected or the script's `REPO_ROOT` calculation must be patched before relying on it. Once corrected, its intended commands are:

```bash
bash scripts/run_generated_cases_serial.sh --dry-run
```

```bash
bash scripts/run_generated_cases_serial.sh \
  --run-root runs/T1/T1_2m_betsy \
  --limit 5
```

Completed and failed cases are marked with:

```text
RUN_COMPLETE
RUN_FAILED
```

The current runner executes cases one at a time, but it is not yet a CIRCE scheduler submission wrapper.

---

## 10. Shared-installation precautions

The installation root contains mutable `DATA` and `OUTPUT_FILES` directories. Running directly in the shared source tree creates several risks:

- one user can replace another user's `Par_file`, `SOURCE`, or `STATIONS`;
- concurrent simulations can overwrite the same output files;
- a failed or partial run can be mistaken for a valid result;
- rebuilding can remove or replace the known working executables; and
- the relationship between inputs and archived outputs can be lost.

Recommended practice:

1. Treat `/shares/seismo_lab/specfem2d/bin` as the shared executable location.
2. Run each model from its own writable case directory.
3. Keep `DATA`, `OUTPUT_FILES`, logs, and metadata together within that case.
4. Do not conduct concurrent runs in the shared source-tree `OUTPUT_FILES` directory.
5. Record the executable checksum, Git commit, MPI module, process count, and complete command with every production suite.
6. Preserve paired cave/no-cave inputs so that the intended model difference is auditable.

---

## 11. Scheduler status

The compilation and initial example were verified, direct MPI commands are known, and CIRCE provides the standard SLURM commands at:

```text
/bin/sbatch
/bin/srun
/bin/sinfo
/bin/squeue
```

On October 7, 2026, `sinfo` showed multiple specialized partitions as available, while the default `general` partition was down. Partition state is dynamic, and the presence of an available partition does not establish that the `seismo_lab` account is authorized to use it. A production SLURM submission procedure has not yet been documented.

The user's SLURM association reported:

```text
Cluster:      circe_slurm
Account:      normal
Partition:    [blank in association output]
Default QOS:  normal
Available QOS: interactive, normal, openaccess, preempt, preempt_short
```

A blank association-level partition indicates that this association is not tied to one named partition. Actual usability still depends on each partition's `AllowQos` setting and current state.

At the time of inspection, the following up partitions accepted the user's default `normal` QOS and allowed all accounts:

| Partition | State | Default time | Preemption mode | Default memory per CPU |
|---|---|---:|---|---:|
| `amd_2021` | UP | 1 hour | OFF | 512 MB |
| `bfbsm_2019` | UP | 1 hour | OFF | 512 MB |
| `cbcs` | UP | 1 hour | REQUEUE | 1,024 MB |
| `snsm_itn19` | UP | 1 hour | OFF | 512 MB |

`amd_2021` is therefore a reasonable provisional choice for a small validation job using `--account=normal` and `--qos=normal`. Partition state and policy can change, so check `sinfo` immediately before submission.

Other intersections exist for `openaccess`, `preempt`, `preempt_short`, and `interactive`, but those should not be selected merely because they are technically available. In particular, preemptible partitions require a restart-safe workflow.

No existing `*.slurm`, `*.sbatch`, or `submit*.sh` files were found within four directory levels of `/shares/seismo_lab`.

Before launching a large survey suite, determine:

- the appropriate CIRCE partition and account;
- wall-time and memory requests;
- whether `mpirun` or `srun` is preferred inside an allocation;
- the permitted relationship between tasks and CPUs per task;
- output and error log locations; and
- whether the OpenMPI MCA workaround remains necessary on the selected compute nodes.

Avoid running long production simulations on an interactive/login node merely because a small validation case succeeds there.

### Provisional four-rank validation script

The following script matches the successful four-rank example and uses the verified MPI-launched mesher. It has not yet been tested through SLURM. Submit it only from an independent case directory containing a complete and internally consistent `DATA` directory; do not use the shared source-tree working directories for concurrent jobs.

An editable copy is provided as [`run_specfem2d_validation.slurm`](run_specfem2d_validation.slurm).

```bash
#!/usr/bin/env bash
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

if [[ "${PAR_NPROC}" != "${NPROC}" ]]; then
  echo "ERROR: DATA/Par_file NPROC=${PAR_NPROC}; SLURM_NTASKS=${NPROC}" >&2
  exit 2
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
```

Before submitting:

```bash
grep -n '^[[:space:]]*NPROC' DATA/Par_file
sinfo -p amd_2021
sbatch --test-only run_specfem2d_validation.slurm
```

Then submit with:

```bash
sbatch run_specfem2d_validation.slurm
```

Monitor it with:

```bash
squeue -j JOB_ID
```

After completion, verify the SLURM output, `logs/meshfem.log`, `logs/specfem.log`, expected seismograms, and the solver's successful-completion message before treating the scheduler workflow as validated.

### Scheduler validation status

On October 7, 2026, CIRCE accepted the provisional script with `sbatch --test-only`:

```text
Job estimate:  34094450
Estimated start: 2026-10-08 08:13:27
Processors:    4
Node:          svc-3024-54-2
Partition:     amd_2021
```

This confirms that the requested account, partition, QOS, processor count, memory, and wall time form a scheduler-valid request. `--test-only` does not submit the job or execute the shell script, so it does **not** yet validate:

- the independent case-directory layout;
- the required `DATA` files and referenced paths;
- the `NPROC` consistency check;
- MPI startup on the allocated compute node;
- meshing and solver execution; or
- output generation and archiving.

The estimated start time is a scheduling estimate, not a reservation or guarantee.

### Build an independent copy of the bundled validation case

The bundled example contains its complete `DATA` directory, including both interface files. It generates its stations internally (`use_existing_STATIONS = .false.`), so a `STATIONS` input is not required for this particular validation case.

Create an independent working copy under `/work`:

```bash
cd /work/t/thompsong
mkdir -p specfem2d_validation_simple
cp -a \
  /shares/seismo_lab/specfem2d/EXAMPLES/simple_topography_and_also_a_simple_fluid_layer/DATA \
  specfem2d_validation_simple/
cp run_specfem2d_validation.slurm specfem2d_validation_simple/
cd specfem2d_validation_simple
```

Change only the copied `Par_file` to four MPI processes:

```bash
sed -i -E \
  's/^([[:space:]]*NPROC[[:space:]]*=[[:space:]]*)[0-9]+/\1 4/' \
  DATA/Par_file
```

Verify the relevant configuration before submission:

```bash
grep -nE '^[[:space:]]*(NPROC|SIMULATION_TYPE|use_existing_STATIONS|interfacesfile)' \
  DATA/Par_file
```

Both interface files listed in the example inventory will be present under the copied `DATA` directory. Confirm that the value of `interfacesfile` resolves from the case directory before submitting.

Then validate the scheduler request again from inside the completed case directory:

```bash
sbatch --test-only run_specfem2d_validation.slurm
```

Only after these checks should the real validation job be submitted.

The independent case was created successfully at:

```text
/work/t/thompsong/specfem2d_validation_simple
```

Its verified settings are:

```text
SIMULATION_TYPE        = 1
NPROC                  = 4
use_existing_STATIONS  = .false.
interfacesfile         = ./interfaces_simple_topo_curved.dat
```

The referenced interface file is included in the copied `DATA` directory. The case is therefore ready for shell-syntax validation and scheduler submission testing.

Shell syntax, interface-file presence, and a second scheduler test all passed. The first real validation job was then submitted:

```text
Job ID:      34094467
Partition:   amd_2021
Job name:    specfem2d-test (abbreviated in default squeue output)
Tasks:       4
Initial state: PD (pending)
Pending reason: Priority
```

The preceding `--test-only` request was scheduler-valid and estimated a start time of October 8, 2026 at 08:13:27. The actual job's start time remains subject to scheduling changes.

---

## 12. Recommended provenance record for every production run

Record the following in a text or JSON file stored with the outputs:

```text
Run identifier
Run start and completion times
CIRCE hostname and job ID
SPECFEM2D Git commit
xmeshfem2D SHA-256 checksum
xspecfem2D SHA-256 checksum
Loaded modules
mpirun version
NPROC
Complete mesher command
Complete solver command
Path or checksum for Par_file
Path or checksum for SOURCE
Path or checksum for STATIONS
Path or checksum for interfaces/model files
Cave/no-cave state
Source position and shot identifier
Exit status
Locations of logs and outputs
```

---

## 13. Additional information still to collect

The exact configure invocation, SPECFEM2D version, OpenMPI wrapper configuration, compiled Scotch support, executable checksums, shared-library dependencies, timestamps, test `Par_file`, and basic SLURM inventory have now been recorded. The following read-only commands would complete the remaining installation and runtime record.

### Remaining compiler and MPI details

```bash
which gcc gfortran mpifort mpirun
gcc --version | head -n 1
gfortran --version | head -n 1
mpifort --version | head -n 1
mpirun --version
```

### Scheduler software version

```bash
sbatch --version
srun --version
```

### Mod15 and Felix-run inventory

After the archived configuration has been copied to CIRCE, inventory it without modifying it:

```bash
find /path/to/Mod15 -maxdepth 3 -type f | sort
```

Particularly valuable files are:

- Sarah's README;
- original and CIRCE-adapted run scripts;
- `Par_file`, `SOURCE`, `SOURCE_template`, and `STATIONS`;
- interface, topography, region, and material files;
- solver and mesher logs;
- archived SU seismograms; and
- output logs reporting successful completion and processor counts.

---

## 14. Remaining validation tasks

1. Correct both survey runners so that `xmeshfem2D` uses `NPROC` MPI ranks for parallel cases.
2. Test and refine the provisional CIRCE scheduler submission script with the bundled example.
3. Reproduce Sarah's Mod15 run without changing its inputs.
4. Confirm that the reproduced outputs agree with Sarah's archived results.
5. Validate one source-only change while reusing the existing mesh.
6. Run a small generated cave/no-cave pair and confirm that metadata, logs, and outputs are complete.
7. Only then launch the full field-survey model suites.
