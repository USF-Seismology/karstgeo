# f-k code review: `spectral.py` vs Sarah's `core_energy.py`

## Main finding

Sarah's function is not doing the same operation as the f-k filtering code in `spectral.py` / `65_compare_gather_pairs`.

Sarah's `windowed_directional_energy()` is a **diagnostic measurement**: it estimates forward and backward directional energy in short moving receiver windows. It does not reconstruct a filtered gather, so it cannot create post-filter ringing, edge wraparound, or apparent reflected arrivals in the time-offset gather.

The `apply_fk_velocity_filter()` / `apply_fk_filter_to_matrix()` code is a **2-D f-k fan filter**: it FFTs the gather in receiver and time, mutes low apparent velocities, then inverse-transforms back to time-offset space. This can create apparent reflection-like artifacts when the spatial FFT assumptions are violated, especially for split-spread gathers, windows that straddle the source, abrupt gather edges, strong direct/ground-roll energy, or nonstationary wavefields.

## Important implementation differences

1. Sarah uses local moving windows and skips windows that straddle the source.

```python
if lo < source_x_m < hi:
    straddles_source[i] = True
    continue
```

This avoids putting left-going and right-going direct arrivals in the same f-k transform window.

2. Sarah analyzes only positive temporal frequencies:

```python
keep = (F > freq_min) & (F <= freq_max) & (K != 0)
```

This makes the sign of `K` easier to interpret for directionality.

3. Sarah classifies forward/backward direction depending on whether the window is left or right of the source:

```python
forward_is_negative_k = xc > source_x_m
```

So the same physical propagation direction is not confused across opposite sides of a split spread.

4. Sarah does not inverse FFT anything. No time-domain reconstruction means no f-k-filter artifacts.

5. The old `spectral.py` global f-k filter applies a single FFT to the whole gather. For split-spread data, this mixes waves propagating away from the source in both directions in one periodic spatial transform. That is the most likely explanation for apparent reflection-like artifacts.

6. The newer `65_compare_gather_pairs_v16/v17` matrix f-k filter is safer because it defaults to `split_at_source=True`, spatial tapering, and receiver-axis zero padding. This is more consistent with Sarah's source-straddling avoidance, but it is still a reconstruction filter, not just an energy diagnostic.

## Practical recommendation

For figures meant to diagnose directional/backscatter energy, use Sarah-style local directional-energy diagnostics.

For producing f-k-filtered gathers for Sarah to load in RefraPy/SeisImager/ReflexW, keep the current split-at-source f-k filter, but label products clearly as **diagnostic f-k filtered gathers** and do not overinterpret weak reflection-like wiggles near the filter boundaries.

## Recommended defaults

For split-spread field gathers:

```bash
--write-fk-filtered \
--fk-min-velocity-mps 500 \
--fk-taper-width-mps 100 \
--fk-use-taper \
--fk-split-at-source \
--fk-spatial-taper-fraction 0.05 \
--fk-pad-factor 2
```

If reflection-like artifacts persist, test these in order:

```bash
--fk-spatial-taper-fraction 0.10
--fk-pad-factor 4
--fk-min-velocity-mps 700
```

and compare against Sarah's `windowed_directional_energy()` maps rather than treating f-k-filtered wiggles alone as evidence for reflections.
