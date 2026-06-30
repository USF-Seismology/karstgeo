# Betsy Wiggle Picker standalone app

This app is a Tk/matplotlib replacement for trying to pick inside RefraPy when RefraPy does not ingest SEG-Y geometry correctly.

## Run

```bash
conda activate flovopy_plus
python betsy_wiggle_picker_app.py
```

Dependencies:

```bash
conda install obspy scipy numpy pandas matplotlib
```

Tkinter must also be available. On most Anaconda Python installs it is already included.

## Input files

By default the app looks in:

```text
/Volumes/tachyon/LBSSP_DATA/betsy_gun_alignment_v1/segy_for_refrapy
```

for the SEG-Y files exported by notebook 99, including:

```text
*_nodal_Z_normalized.sgy
*_nodal_N_normalized.sgy
*_nodal_E_normalized.sgy
*_geode_shifted_normalized.sgy
*_combined_nodal_geode_shifted_normalized.sgy
```

You can also load files manually with the buttons on the left.

## Controls

Display modes:

```text
nodal_Z
nodal_N
nodal_E
geode
nodal_Z_plus_geode
combined
```

Processing controls:

```text
detrend on/off
taper percentage
filter on/off
freqmin
freqmax
corners
```

Display controls:

```text
gain
trace scale
clip %
time min/max
shade: none, positive, negative, both
```

## Picking

Pick mode:

```text
left-click  = add pick at nearest receiver
right-click = delete nearest pick
```

Line mode:

```text
left-click two points = draw velocity line
right-click = delete last line or pending point
```

Velocity is estimated as:

```text
v = dx / dt
```

The displayed label uses absolute velocity.

In overlay mode:

```text
left-click   = nodal Z pick/line point
middle-click = Geode pick/line point
```

## Outputs

Click `Save picks/lines` to write:

```text
betsy_interactive_picks.csv
betsy_interactive_velocity_lines.csv
```
