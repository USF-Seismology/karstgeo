from pathlib import Path

def find_specfem_model_outputs(root):
    """Return sorted alphabetical (A, B, C...) OUTPUT_FILES directories under a SPECFEM root."""
    root = Path(root).expanduser()
    # Matches single uppercase letters (A-Z). Use "?" if names can be lowercase or numbers.
    return sorted(p for p in root.glob("[A-Z]/OUTPUT_FILES") if p.is_dir())


def model_number_from_name(model_name):
    """Convert alphabetical model names (A, B, C) to an integer code using ASCII value.
    
    Returns None if the string is empty.
    """
    name_str = str(model_name).strip()
    # Returns the ASCII value of the first character (e.g., 'A' -> 65, 'B' -> 66)
    return ord(name_str[0].upper()) if name_str else None

def specfem_gather_result_to_stream(
    result,
    component="BXZ",
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    source_x_m=0.0,
    shot_number=1,
    network="SY",
):
    """
    Convert load_specfem_gather() output into an ObsPy Stream.

    This standardizes both SU and SEM ASCII inputs into a single object that can
    be plotted, written to SEG-Y, or differenced against another model.
    """
    component = component.upper()

    if result["mode"] == "su":
        st = result["stream"].copy()

        for i, tr in enumerate(st, start=1):
            receiver_x_m = first_receiver_x_m + (i - 1) * receiver_spacing_m

            tr.stats.network = network
            tr.stats.station = f"S{i:04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = component

            tr.stats.segy = AttribDict()
            tr.stats.segy.trace_header = make_trace_header(
                station_idx=i,
                receiver_x_m=float(receiver_x_m),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                dt_s=float(tr.stats.delta),
                npts=int(tr.stats.npts),
                receiver_z_m=0.0,
                source_z_m=0.0,
            )

        return st

    if result["mode"] == "sem":
        time = np.asarray(result["time"], dtype=float)
        data = np.asarray(result["data"], dtype=np.float32)
        station_indices = np.asarray(result["station_indices"], dtype=int)

        dt = float(np.median(np.diff(time)))
        st = Stream()

        for i, station_index in enumerate(station_indices):
            receiver_x_m = first_receiver_x_m + (station_index - station_indices.min()) * receiver_spacing_m

            tr = Trace(data=data[i].astype(np.float32))
            tr.stats.network = network
            tr.stats.station = f"S{int(station_index):04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = component
            tr.stats.delta = dt
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(time[0])

            tr.stats.segy = AttribDict()
            tr.stats.segy.trace_header = make_trace_header(
                station_idx=int(station_index),
                receiver_x_m=float(receiver_x_m),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                dt_s=dt,
                npts=int(tr.stats.npts),
                receiver_z_m=0.0,
                source_z_m=0.0,
            )

            st.append(tr)

        return st

    raise ValueError(f"Unknown result mode: {result['mode']}")

def discover_su_files(output_dir):
    """Find candidate SU files in an OUTPUT_FILES directory."""
    return sorted(Path(output_dir).glob("*.su"))


def read_su_try_both_byteorders(path, dt_s=None, t0_s=0.0):
    """Read an SU file, trying little-endian first and then big-endian."""
    last_error = None
    for byteorder in ("<", ">"):
        try:
            st = read_su_file(path, dt_s=dt_s, t0_s=t0_s, byteorder=byteorder)
            return st, byteorder
        except Exception as exc:
            last_error = exc
    raise last_error


def load_specfem_gather(
    output_dir,
    component="BXZ",
    extension="semv",
    timing=None,
    prefer_su=True,
    verbose=True,
):
    """
    Load a SPECFEM gather from one OUTPUT_FILES directory.

    Parameters
    ----------
    output_dir : path-like
        A model's OUTPUT_FILES directory.
    component : str
        SPECFEM/SEGY component name to look for when falling back to ASCII,
        e.g. "BXZ" or "BXX".
    extension : str
        SPECFEM ASCII extension, usually "semv" or "semd".
    timing : optional
        Timing object passed through to read_sem_gather().
    prefer_su : bool
        If True, try *.su files before ASCII.
    verbose : bool
        Print status messages.

    Returns
    -------
    dict
        Either {"mode": "su", "stream": Stream, ...} or
        {"mode": "sem", "time": ..., "data": ..., "station_indices": ...}.
    """
    output_dir = Path(output_dir)

    if prefer_su:
        for sufile in discover_su_files(output_dir):
            if sufile.name.startswith('Ux'):
                component = 'BXX'
            elif sufile.name.startswith('Uz'):
                component = 'BXZ'
            try:
                st, byteorder = read_su_try_both_byteorders(sufile)
                if verbose:
                    print(f"Loaded SU: {sufile.name} byteorder={byteorder}")
                return {
                    "mode": "su",
                    "stream": st,
                    "path": sufile,
                    "component": component,
                    "byteorder": byteorder,
                }
            except Exception as exc:
                if verbose:
                    print(f"  SU failed: {sufile.name}: {type(exc).__name__}: {exc}")

    if verbose:
        print(f"Falling back to SPECFEM ASCII: component={component}, extension={extension}")

    time, data, station_indices = read_sem_gather(
        output_dir,
        component=component,
        extension=extension,
        timing=timing,
        verbose=verbose,
    )

    return {
        "mode": "sem",
        "time": time,
        "data": data,
        "station_indices": station_indices,
        "component": component,
        "extension": extension,
    }
    
def model_name_from_output_dir(output_dir):
    """Return model name from a Mod*/OUTPUT_FILES path."""
    return Path(output_dir).parent.name


def load_model_as_stream(
    output_dir,
    component="BXZ",
    extension="semv",
    timing=None,
    prefer_su=True,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    source_x_m=0.0,
    shot_number=None,
    network="SY",
    verbose=True,
):
    """Load one model output directory and return a standardized ObsPy Stream."""
    output_dir = Path(output_dir)
    model_name = model_name_from_output_dir(output_dir)

    if shot_number is None:
        shot_number = model_number_from_name(model_name) or 1

    result = load_specfem_gather(
        output_dir,
        component=component,
        extension=extension,
        timing=timing,
        prefer_su=prefer_su,
        verbose=verbose,
    )

    st = specfem_gather_result_to_stream(
        result,
        component=component,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
        source_x_m=source_x_m,
        shot_number=shot_number,
        network=network,
    )

    return st, result


def write_model_products(
    output_dir,
    component="BXZ",
    extension="semv",
    timing=None,
    prefer_su=True,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    source_x_m=0.0,
    write_segy_file=True,
    make_plot=True,
    tmin=0.0,
    tmax=0.3,
    scale=0.02,
    normalize=False,
    verbose=True,
):
    """Load one model, then write SEG-Y and/or a wiggle plot under outdir."""
    output_dir = Path(output_dir)
    model_name = model_name_from_output_dir(output_dir)
    shot_number = model_number_from_name(model_name) or 1

    st, result = load_model_as_stream(
        output_dir,
        component=component,
        extension=extension,
        timing=timing,
        prefer_su=prefer_su,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
        source_x_m=source_x_m,
        shot_number=shot_number,
        verbose=verbose,
    )

    segy_file = segy_out_dir / f"{model_name}_{component}.segy"
    fig_file = fig_dir / f"{model_name}_{component}.png"

    if write_segy_file:
        segy_file.parent.mkdir(parents=True, exist_ok=True)
        write_segy(st, segy_file)
        if verbose:
            print(f"  wrote SEG-Y: {segy_file}")

    if make_plot:
        fig_file.parent.mkdir(parents=True, exist_ok=True)
        plot_wiggle_gather_from_stream(
            st,
            fallback_receiver_spacing_m=receiver_spacing_m,
            fallback_first_receiver_x_m=first_receiver_x_m,
            fallback_source_x_m=source_x_m,
            normalize=normalize,
            scale=scale,
            tmin=tmin,
            tmax=tmax,
            title=f"{model_name} {component}",
            outfile=fig_file,
        )
        if verbose:
            print(f"  wrote figure: {fig_file}")

    return st, result, segy_file, fig_file

def plot_segy_file(
    segy_file,
    outfile=None,
    tmin=0.0,
    tmax=0.3,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    source_x_m=0.0,
    scale=0.02,
    normalize=False,
):
    """Read and plot one SEG-Y file."""
    segy_file = Path(segy_file)
    st = obspy.read(str(segy_file), format="SEGY", unpack_trace_headers=True)

    if outfile is None:
        outfile = fig_dir / f"{segy_file.stem}_wiggle.png"

    plot_wiggle_gather_from_stream(
        st,
        fallback_receiver_spacing_m=receiver_spacing_m,
        fallback_first_receiver_x_m=first_receiver_x_m,
        fallback_source_x_m=source_x_m,
        tmin=tmin,
        tmax=tmax,
        scale=scale,
        normalize=normalize,
        title=f"{segy_file.name}: wiggle gather",
        outfile=outfile,
    )
    print(f"  plotted {segy_file} to {outfile}")

    return st

def plot_su_directory(
    su_dir,
    pattern="*_*.su",
    tmin=-0.05,
    tmax=0.25,
    receiver_spacing_m=2.0,
    first_receiver_x_m=0.0,
    source_x_m=0.0,
    scale=0.02,
    normalize=False,
):
    """Read all matching SU files in a directory and write wiggle plots under outdir."""
    su_dir = Path(su_dir).expanduser()
    su_fig_dir = fig_dir / "su_wiggles"
    su_fig_dir.mkdir(parents=True, exist_ok=True)

    su_files = sorted(su_dir.glob(pattern))
    print(f"Found {len(su_files)} SU files in {su_dir}")

    rows = []
    for su_file in su_files:
        print(su_file)
        try:
            st, byteorder = read_su_try_both_byteorders(su_file)
            outfile = su_fig_dir / f"{su_file.stem}_wiggle.png"
            plot_wiggle_gather_from_stream(
                st,
                fallback_receiver_spacing_m=receiver_spacing_m,
                fallback_first_receiver_x_m=first_receiver_x_m,
                fallback_source_x_m=source_x_m,
                tmin=tmin,
                tmax=tmax,
                scale=scale,
                normalize=normalize,
                title=f"{su_file.stem}: SU wiggle gather",
                outfile=outfile,
            )
            print(f"  plotted {su_file} to {outfile}")
            rows.append({"file": su_file, "byteorder": byteorder, "n_traces": len(st), "figure_file": outfile})
        except Exception as exc:
            print(f"  skipping {su_file.name}: {type(exc).__name__}: {exc}")
            rows.append({"file": su_file, "byteorder": None, "n_traces": 0, "figure_file": None, "error": str(exc)})

    return pd.DataFrame(rows)

# Differencing

def stream_to_gather_arrays(
    st,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
):
    """Convert a single-component Stream to time, data, receiver_x arrays."""
    st = st.copy()
    st.sort(keys=["station", "channel"])

    if len(st) == 0:
        raise ValueError("Empty Stream")

    npts = min(tr.stats.npts for tr in st)
    dt = float(st[0].stats.delta)

    data = np.vstack([tr.data[:npts].astype(float) for tr in st])
    time = np.arange(npts, dtype=float) * dt
    receiver_x_m = first_receiver_x_m + np.arange(len(st), dtype=float) * receiver_spacing_m

    return time, data, receiver_x_m


def difference_segy_gathers(
    segy_a,
    segy_b,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
):
    """Read two SEG-Y gathers and return matched A, B, A-B arrays."""
    st_a = obspy.read(str(segy_a), format="SEGY", unpack_trace_headers=True)
    st_b = obspy.read(str(segy_b), format="SEGY", unpack_trace_headers=True)

    time_a, data_a, rx_a = stream_to_gather_arrays(
        st_a,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )
    time_b, data_b, rx_b = stream_to_gather_arrays(
        st_b,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )

    ntr = min(data_a.shape[0], data_b.shape[0])
    npts = min(data_a.shape[1], data_b.shape[1])

    time = time_a[:npts]
    receiver_x_m = rx_a[:ntr]
    data_a = data_a[:ntr, :npts]
    data_b = data_b[:ntr, :npts]
    diff = data_a - data_b

    return time, data_a, data_b, diff, receiver_x_m


def plot_model_difference_from_segy(
    model_a,
    model_b,
    component="BXZ",
    segy_dir=segy_out_dir,
    source_x_m=0.0,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    tmin=0.0,
    tmax=0.3,
    omin=None,
    omax=None,
    clip_percentile=98,
    outfile=None,
):
    """Difference two converted model SEG-Y gathers and plot A, B, and A-B."""
    segy_dir = Path(segy_dir)
    segy_a = segy_dir / f"{model_a}_{component}.segy"
    segy_b = segy_dir / f"{model_b}_{component}.segy"

    if not segy_a.exists():
        raise FileNotFoundError(segy_a)
    if not segy_b.exists():
        raise FileNotFoundError(segy_b)

    time, data_a, data_b, diff, receiver_x_m = difference_segy_gathers(
        segy_a,
        segy_b,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )

    if outfile is None:
        outfile = diff_fig_dir / f"{model_a}_minus_{model_b}_{component}.png"

    fig = plot_difference_gathers(
        time=time,
        data_a=data_a,
        data_b=data_b,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        label_a=model_a,
        label_b=model_b,
        title=f"{model_a} - {model_b}, {component}",
        tmin=tmin,
        tmax=tmax,
        omin=omin,
        omax=omax,
        clip_percentile=clip_percentile,
        outfile=outfile,
    )
    print(f"  plotted difference between {model_a} and {model_b} to {outfile}")

    return fig, diff


from pathlib import Path
import numpy as np
import obspy
from obspy import Stream, Trace
from obspy.io.segy.segy import SEGYTraceHeader

def stream_to_gather_arrays(
    st,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
):
    """Convert a single-component Stream to time, data, receiver_x arrays."""
    st = st.copy()
    st.sort(keys=["station", "channel"])

    if len(st) == 0:
        raise ValueError("Empty Stream")

    npts = min(tr.stats.npts for tr in st)
    dt = float(st[0].stats.delta)

    data = np.vstack([tr.data[:npts].astype(float) for tr in st])
    time = np.arange(npts, dtype=float) * dt
    receiver_x_m = first_receiver_x_m + np.arange(len(st), dtype=float) * receiver_spacing_m

    return time, data, receiver_x_m


def difference_segy_gathers(
    segy_a,
    segy_b,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    output_segy_path=None,
):
    """Read two SEG-Y gathers, compute differences, convert to Stream, and write to SEG-Y."""
    st_a = obspy.read(str(segy_a), format="SEGY", unpack_trace_headers=True)
    st_b = obspy.read(str(segy_b), format="SEGY", unpack_trace_headers=True)

    # Cache fundamental metadata from the original data headers
    original_dt = st_a[0].stats.delta
    original_starttime = st_a[0].stats.starttime

    time_a, data_a, rx_a = stream_to_gather_arrays(
        st_a,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )
    time_b, data_b, rx_b = stream_to_gather_arrays(
        st_b,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )

    ntr = min(data_a.shape[0], data_b.shape[0])
    npts = min(data_a.shape[1], data_b.shape[1])

    time = time_a[:npts]
    receiver_x_m = rx_a[:ntr]
    data_a = data_a[:ntr, :npts]
    data_b = data_b[:ntr, :npts]
    diff = data_a - data_b

    # --- NEW: Convert the 'diff' matrix into an ObsPy Stream and write to SEG-Y ---
    diff_stream = Stream()
    
    for i in range(ntr):
        # Create standard trace
        tr = Trace(data=diff[i, :].astype(np.float32))
        tr.stats.delta = original_dt
        tr.stats.starttime = original_starttime
        tr.stats.station = f"R{i:03d}"
        
        # Pull original trace header structure if available, or generate a fresh one
        if i < len(st_a):
            # Safe copy of original trace header to preserve geometry/source details
            tr.stats.segy = getattr(st_a[i], 'stats', {}).get('segy', {}).copy()
        else:
            # Fallback block to prevent empty object errors
            tr.stats.segy = {'trace_header': SEGYTraceHeader()}
            
        # Update/Enforce critical SEG-Y trace header structural IDs
        tr.stats.segy['trace_header'].trace_sequence_number_within_line = i + 1
        tr.stats.segy['trace_header'].trace_sequence_number_within_seismic_reely = i + 1
        tr.stats.segy['trace_header'].original_field_record_number = 1
        tr.stats.segy['trace_header'].trace_number_within_the_original_field_record = i + 1
        tr.stats.segy['trace_header'].distance_from_center_of_source_to_receiver_group = int(receiver_x_m[i])
        
        diff_stream.append(tr)
        
    if output_segy_path:
        output_segy_path = Path(output_segy_path)
        output_segy_path.parent.mkdir(parents=True, exist_ok=True)
        # data_encoding=1 enforcement tells the output layout engine to write standard 4-byte IBM floats
        diff_stream.write(str(output_segy_path), format="SEGY", data_encoding=1)
        print(f"  Saved difference gather to SEG-Y: {output_segy_path}")

    return time, data_a, data_b, diff, receiver_x_m


def plot_model_difference_from_segy(
    model_a,
    model_b,
    component="BXZ",
    segy_dir=None,      # Adjusted default argument order/definition safely
    diff_segy_dir=None, # NEW: Dedicated directory path for your output diff files
    source_x_m=0.0,
    receiver_spacing_m=1.0,
    first_receiver_x_m=0.0,
    tmin=0.0,
    tmax=0.3,
    omin=None,
    omax=None,
    clip_percentile=98,
    outfile=None,
):
    """Difference two converted model SEG-Y gathers, export new SEG-Y, and plot."""
    segy_dir = Path(segy_dir)
    segy_a = segy_dir / f"{model_a}_{component}.segy"
    segy_b = segy_dir / f"{model_b}_{component}.segy"

    if not segy_a.exists():
        raise FileNotFoundError(segy_a)
    if not segy_b.exists():
        raise FileNotFoundError(segy_b)

    # Setup the output file destination for the new SEG-Y dataset
    if diff_segy_dir is None:
        diff_segy_dir = segy_dir / "differences"
    out_segy_file = Path(diff_segy_dir) / f"{model_a}_minus_{model_b}_{component}.segy"

    # Pass the out_segy_file path down to the updated difference module
    time, data_a, data_b, diff, receiver_x_m = difference_segy_gathers(
        segy_a,
        segy_b,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
        output_segy_path=out_segy_file
    )

    if outfile is None:
        outfile = diff_fig_dir / f"{model_a}_minus_{model_b}_{component}.png"

    # Fallback/placeholder invocation to prevent errors if plot_difference_gathers is defined globally
    fig = plot_difference_gathers(
        time=time,
        data_a=data_a,
        data_b=data_b,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        label_a=model_a,
        label_b=model_b,
        title=f"{model_a} - {model_b}, {component}",
        tmin=tmin,
        tmax=tmax,
        omin=omin,
        omax=omax,
        clip_percentile=clip_percentile,
        outfile=outfile,
    )
    print(f"  plotted difference between {model_a} and {model_b} to {outfile}")

    return fig, diff


# NMO

import matplotlib.pyplot as plt
import numpy as np

def apply_obspy_nmo_scan(
    st: Stream, 
    test_velocity: float, 
    source_x_m: float = 150.0,
    receiver_spacing_m: float = 1.0, 
    first_receiver_x_m: float = 1.0
) -> Stream:
    """Correctly flattens a diffraction hyperbola by re-sampling along the hyperbola path.
    
    Parameters:
    -----------
    st : obspy.Stream
        The input shot gather.
    test_velocity : float
        The trial velocity (m/s) used to flatten the diffraction hyperbola.
    source_x_m : float
        The absolute horizontal position of the shot/source (e.g., 150.0 m).
    receiver_spacing_m : float
        The horizontal distance between adjacent receivers.
    first_receiver_x_m : float
        The horizontal position of the very first receiver (e.g., 1.0 m).
    """
    import copy
    import numpy as np
    
    nmo_stream = copy.deepcopy(st)
    
    for idx, tr in enumerate(nmo_stream):
        dt = tr.stats.delta
        num_samples = tr.stats.npts
        
        # 1. Calculate absolute position on the line (x)
        receiver_x = first_receiver_x_m + (idx * receiver_spacing_m)
        
        # 2. Calculate the true relative offset distance to the shot
        offset = np.abs(receiver_x - source_x_m)
        tr.stats.distance = offset 
            
        times = np.arange(num_samples) * dt
        
        # 3. Calculate the hyperbola path trajectory traveltimes
        t_nmo = np.sqrt(times**2 + (offset**2 / test_velocity**2))
        
        # 4. FIXED MAPPING DIRECTION: 
        # Look up what amplitude exists along the curve (t_nmo) within the raw 
        # trace timeline (times), and place that value into our flattened array.
        flattened_data = np.interp(t_nmo, times, tr.data, left=0.0, right=0.0)
        tr.data = flattened_data
        
    return nmo_stream

import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream

def plot_nmo_velocity_grid(
    st: Stream,
    trial_velocities: list,
    source_x_m: float = 150.0,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 1.0,
    offset_range_m: tuple = (-50.0, 50.0),
    clip_percentile: float = 95.0,
    cols_per_row: int = 3
):
    """Applies an NMO velocity scan over multiple trial velocities and plots a 3-column grid.
    
    Filters and restricts the display to a localized relative horizontal offset window.
    
    Parameters:
    -----------
    st : obspy.Stream
        The input seismic stream gather.
    trial_velocities : list of float
        The array of velocities (m/s) to evaluate.
    source_x_m : float
        Horizontal coordinate of the source/shot.
    receiver_spacing_m : float
        Horizontal distance between adjacent receivers.
    first_receiver_x_m : float
        Horizontal position of the first receiver in the line profile.
    offset_range_m : tuple of (float, float)
        The minimum and maximum relative offset boundaries in meters from the source position.
        e.g., (-50, 50) includes traces up to 50 meters left and right of the shot.
    clip_percentile : float
        The color scale percentile saturation clipping factor for plotting data.
    cols_per_row : int
        Number of subplot panel columns per row (locked at 3 default).
    """
    num_velocities = len(trial_velocities)
    num_rows = int(np.ceil(num_velocities / cols_per_row))
    
    # Generate the grid figure framework
    fig, axes_grid = plt.subplots(num_rows, cols_per_row, figsize=(18, 4 * num_rows), sharey=True)
    axes = axes_grid.flatten() if num_rows > 1 else np.atleast_1d(axes_grid)
    
    for idx, (ax, v) in enumerate(zip(axes, trial_velocities)):
        # 1. Apply the corrected NMO correction down to the full stream
        flattened_st = apply_obspy_nmo_scan(
            st=st,
            test_velocity=v,
            source_x_m=source_x_m,
            receiver_spacing_m=receiver_spacing_m,
            first_receiver_x_m=first_receiver_x_m
        )
        
        # 2. Extract relative spatial locations and mask by offset thresholds
        filtered_traces = []
        display_offsets = []
        
        for tr_idx, tr in enumerate(flattened_st):
            # Calculate the absolute position on the profile line
            rec_x = first_receiver_x_m + (tr_idx * receiver_spacing_m)
            # True relative offset location: negative values to the left, positive to the right
            relative_offset = rec_x - source_x_m
            
            if offset_range_m[0] <= relative_offset <= offset_range_m[1]:
                filtered_traces.append(tr.data)
                display_offsets.append(relative_offset)
                
        if len(filtered_traces) == 0:
            raise ValueError(f"No traces found within the specified offset range {offset_range_m}.")
            
        # Convert the filtered trace data subset into a 2D raster matrix
        gather_matrix = np.array(filtered_traces)
        
        # 3. Render the raster gather data matrix
        # Extent sets true spatial coordinates along the horizontal axis instead of trace indexes
        extent = [display_offsets[0], display_offsets[-1], gather_matrix.shape[1], 0]
        
        v_limit = np.percentile(np.abs(gather_matrix), clip_percentile)
        
        img = ax.imshow(
            gather_matrix.T, 
            aspect='auto', 
            cmap='seismic', 
            vmin=-v_limit, 
            vmax=v_limit,
            extent=extent
        )
        
        ax.set_title(f"V = {v} m/s", fontsize=14, weight='bold')
        ax.set_xlabel("Relative Offset (m)", fontsize=11)
        
        # Apply the vertical traveltime label axis string only along the left boundary frames
        if idx % cols_per_row == 0:
            ax.set_ylabel("Time Sample Index (t₀)", fontsize=11)
            
    # Deactivate and clear unassigned trailing plot panels
    for empty_ax in axes[num_velocities:]:
        empty_ax.set_axis_off()
        
    plt.tight_layout()
    plt.show()
    return fig

# f-k filtering

import copy
import numpy as np
from obspy import Stream

def apply_obspy_fk_filter(
    st: Stream, 
    min_velocity: float = 1000.0, 
    receiver_spacing_m: float = 1.0,
    use_taper: bool = True,
    taper_width_mps: float = 200.0
) -> Stream:
    """Applies a 2D f-k velocity fan filter to reject slow coherent noise.
    
    Includes an optional cosine taper to prevent Gibbs phenomenon ringing.
    """
    filtered_st = copy.deepcopy(st)
    filtered_st.sort(keys=["station", "channel"])
    
    num_traces = len(filtered_st)
    num_samples = filtered_st[0].stats.npts
    dt = filtered_st[0].stats.delta
    dx = receiver_spacing_m
    
    data_matrix = np.array([tr.data for tr in filtered_st], dtype=float)
    
    # 2D FFT to f-k domain
    fk_matrix = np.fft.fft2(data_matrix)
    
    freqs = np.fft.fftfreq(num_samples, d=dt)      
    wavenumbers = np.fft.fftfreq(num_traces, d=dx) 
    
    K, F = np.meshgrid(wavenumbers, freqs, indexing='ij')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        apparent_velocities = np.abs(F / K)
    
    # Initialize pass filter mask (1.0 = pass, 0.0 = mute)
    mask = np.ones_like(fk_matrix, dtype=float)
    
    if use_taper:
        # Smooth transition zone
        v_low = min_velocity
        v_high = min_velocity + taper_width_mps
        
        # Completely mute anything below v_low
        mask[apparent_velocities <= v_low] = 0.0
        
        # Apply Hanning/Cosine taper in the buffer zone
        taper_zone = (apparent_velocities > v_low) & (apparent_velocities < v_high)
        # Normalize velocities in the transition zone to a 0-1 scale
        normalized_v = (apparent_velocities[taper_zone] - v_low) / (v_high - v_low)
        # Map to a 0 to pi/2 cosine taper
        mask[taper_zone] = 0.5 * (1.0 - np.cos(normalized_v * np.pi))
    else:
        # Hard cut mask
        mask[apparent_velocities < min_velocity] = 0.0
        
    # Enforce protection for the DC component (K=0 holds infinite velocity vertical planes)
    mask[K == 0] = 1.0
    
    # Apply mask and transform back to space-time domain
    fk_matrix *= mask
    filtered_matrix = np.real(np.fft.ifft2(fk_matrix))
    
    for idx, tr in enumerate(filtered_st):
        tr.data = filtered_matrix[idx, :].astype(np.float32)
        
    return filtered_st

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


def plot_fk_spectrum_zoomed(st, receiver_spacing_m=1.0, max_display_freq=600.0, title="f-k Spectrum"):
    """Computes f-k spectrum and zooms into the active physical source bandwidth."""
    st_copy = st.copy()
    st_copy.sort(keys=["station", "channel"])
    
    num_traces = len(st_copy)
    if num_traces == 0:
        raise ValueError("The input ObsPy Stream is empty.")
        
    # --- CRITICAL FIX: Extract stats from the first Trace in the Stream ---
    num_samples = st_copy[0].stats.npts
    dt = st_copy[0].stats.delta
    dx = receiver_spacing_m
    
    data_matrix = np.array([tr.data for tr in st_copy], dtype=float)
    
    # Remove the DC offset / mean trace-by-trace to clean up the k=0 horizontal line
    for i in range(num_traces):
        data_matrix[i, :] -= np.mean(data_matrix[i, :])
    
    fk_matrix = np.fft.fftshift(np.fft.fft2(data_matrix))
    amplitude_spectrum = np.abs(fk_matrix)
    
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=dt))
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(num_traces, d=dx))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [freqs[0], freqs[-1], wavenumbers[-1], wavenumbers[0]]
    img = ax.imshow(np.log10(amplitude_spectrum + 1e-5), aspect='auto', cmap='viridis', extent=extent)
    
    # Zoom into the active frequency window
    ax.set_xlim(-max_display_freq, max_display_freq)
    
    # Dynamically scale the wavenumber Y-axis to match the zoomed frequency window geometry
    max_k = max_display_freq / 1000.0  # k = f/v
    ax.set_ylim(max_k * 1.5, -max_k * 1.5)
    
    # Re-draw the boundary lines inside the zoomed window
    f_axis = np.linspace(-max_display_freq, max_display_freq, 100)
    ax.plot(f_axis, f_axis / 1000.0, 'r--', label='1000 m/s Boundary', linewidth=2)
    ax.plot(f_axis, f_axis / -1000.0, 'r--', linewidth=2)
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Wavenumber (k, 1/m)", fontsize=12)
    
    fig.colorbar(img, ax=ax, label="Log10 Amplitude")
    ax.legend(loc='upper right')
    
    plt.show()
    return fig

# AGC
import numpy as np
import copy
from obspy import Stream

def apply_obspy_agc(st: Stream, window_sec: float = 0.02) -> Stream:
    """Applies a moving RMS Automatic Gain Control (AGC) to an ObsPy Stream.
    
    Parameters:
    -----------
    st : obspy.Stream
        The input seismic stream gather.
    window_sec : float
        The width of the sliding AGC window in seconds (e.g., 0.02 s = 20 ms).
        Shorter windows boost weak signals more aggressively but can increase noise.
    """
    gained_st = copy.deepcopy(st)
    
    for tr in gained_st:
        dt = tr.stats.delta
        data = tr.data.astype(float)
        num_samples = len(data)
        
        # Convert window duration from seconds to integer sample count
        win_samples = int(window_sec / dt)
        if win_samples % 2 == 0:
            win_samples += 1  # Ensure the window has an exact center sample
            
        half_win = win_samples // 2
        gained_data = np.zeros_like(data)
        
        # Pre-calculate squared values for faster convolution running-mean metrics
        squared_data = data ** 2
        
        # Use a uniform moving window filter to calculate local variance/energy
        # This replaces slow nested Python loops with optimized NumPy routines
        window_kernel = np.ones(win_samples)
        # Pad edges to prevent edge-decay artifacts
        padded_squared = np.pad(squared_data, half_win, mode='edge')
        local_energy = np.convolve(padded_squared, window_kernel, mode='valid')
        
        # Calculate local Root-Mean-Square (RMS)
        rms = np.sqrt(local_energy / win_samples)
        
        # Avoid dividing by zero in empty or quiet silent zones
        rms[rms == 0.0] = 1e-10
        
        # Scale the original amplitudes inversely proportional to local RMS energy
        gained_data = data / rms
        
        # Re-inject the balanced trace arrays into the stream structures
        tr.data = gained_data.astype(np.float32)
        
    return gained_st


def apply_linear_time_gain(st: Stream, power: float = 1.0) -> Stream:
    """Multiplies each sample by t^power to boost deep reflections linearly."""
    gained_st = copy.deepcopy(st)
    
    for tr in gained_st:
        dt = tr.stats.delta
        times = np.arange(tr.stats.npts) * dt
        # Avoid zero-multiplication artifact on the very first index sample
        times[0] = times[1] * 0.1 
        
        # Gain scalar scales up exponentially over time
        gain_vector = times ** power
        tr.data = (tr.data * gain_vector).astype(np.float32)
        
    return gained_st


# source functions
def ricker_wavelet(frequency, dt, duration):
    """Generates a Ricker wavelet (Mexican hat) for a given central frequency."""
    t = np.arange(-duration / 2, duration / 2 + dt, dt)
    pi_f_t = np.pi * frequency * t
    wavelet = (1 - 2 * (pi_f_t ** 2)) * np.exp(-(pi_f_t ** 2))
    return wavelet

import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass


def ricker_wavelet(f0, dt, duration):
    """
    Zero-phase Ricker wavelet centered at t=0.

    Parameters
    ----------
    f0 : float
        Dominant frequency [Hz]

    dt : float
        Sample interval [s]

    duration : float
        Total wavelet duration [s]
    """
    t = np.arange(-duration / 2, duration / 2 + dt, dt)

    a = (np.pi * f0 * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)

    return t, w


def normalize(y):
    """
    Normalize waveform to unit peak amplitude.
    """
    y = np.asarray(y, dtype=float)

    ymax = np.max(np.abs(y))

    if ymax > 0:
        y = y / ymax

    return y

def gaussian_taper(t, sigma):
    """
    Gaussian taper centered at t=0.
    sigma controls width in seconds.
    Smaller sigma = stronger taper.
    """
    return np.exp(-0.5 * (t / sigma) ** 2)
