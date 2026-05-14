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
