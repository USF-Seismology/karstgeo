<h2>Overview</h2>
This is a private GitHub repository for all codes and text-based files related to the Karst Geophysics project.
Do not put binary files or documents in here - keep those in the associated Box folder: 
https://usf.box.com/s/b716i8oe4ry22uvun25s6bs9m6q2plyh

Only Jochen, Felix, Sarah, Mel, Rocco, Glenn, Pati, and Mike McNair should have access currently.

Add a README to each folder too please.

<h2>Seismic forward modeling</h2>

- Deepwave (Glenn): https://ausargeo.com/deepwave/
- PyGimLi (Jochen): https://www.pygimli.org/_examples_auto/2_seismics/index.html
- SpecFEM2D (Sarah): https://github.com/SPECFEM/specfem2d


For refraction/diffraction analysis vs MASW. Same shot gathers, two processing branches:

Raw shot gathers
    ├── Refraction / diffraction branch
    │       mute or suppress surface waves
    │       preserve first breaks, refractions, diffractions
    │       pick arrivals
    │       invert with PyGIMLi / RefraPy
    │
    └── MASW branch
            preserve surface waves
            mute/suppress body waves if needed
            transform to f-k or phase-velocity/frequency domain
            pick dispersion curves
            invert for Vs

For refraction, surface waves are mostly a nuisance because they dominate amplitude and confuse first-break pickers. Use f-k filtering, velocity mutes, or time-offset mutes to reduce slow coherent ground roll.

For MASW, surface waves are the signal. You would instead keep the coherent Rayleigh-wave train and suppress early body-wave arrivals, random noise, and late scattered energy.

So the report language could be:

We will treat the shot gathers using two complementary processing branches. For seismic refraction and diffraction analysis, the dominant low-velocity surface-wave energy will be muted or suppressed to improve first-break picking and highlight body-wave arrivals and scattered phases. In parallel, the same surface-wave energy will be retained for MASW analysis, where frequency-dependent surface-wave velocity is used to estimate shallow shear-wave structure. This dual-use workflow allows the same field acquisition to support both travel-time tomography and surface-wave characterization of shallow void-related anomalies.


# TO-DO, July 1, 2026:
Draft Report submitted June 29, 2026

All nodal field data and analysis now in Box

Main thing still to do is to construct combined shot gathers for all nodal events at same shot location. And then add in geode data from refraction and streamer surveys at same shot positions. 
Then pick times. And create better layered seismic velocity models.
Can either do this interactively - which we can partly do with segy_wiggle_picker.py in apps, and we were starting to flesh out much further with commonshot.py and test_commonshot.py and layer on a more comprehensive GUI.
or we can do this automatically, and I already ran my consensus picker - although how successfully i have not measured.
Or I could do interactive picking directly in Antelope, as a backup if I cannot get GUI to work. Or run autodetection. Indeed, that would be a useful demo to students for Antelope.

All paths should be changed to Box.

Deprecated codes should probably be removed.

Also we can make use of many analysis products, like pick times and single shot gather files, and even existing stacked shotgathers that match geode events. Rather than have to rebuild and re-run the processing sequence from scratch around the versions of specfem_tools and segy_tools I refactored in past 2 days.

Felix is running all shot positions for the T1 line, to enable comparison/differencing. But i do need to still find a way to match amplitude between synthetic and real data, so i can do subtraction. But not really worth it without updating velocity models first, which would require a whole new SPECFEM run afterwards anyway.

So the priority is to pick the real data in a better way, to get a better model, rather than do more differencing yet.