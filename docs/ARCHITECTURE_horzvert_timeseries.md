# Architecture: horzvert_timeseries.py

This document describes how `horzvert_timeseries.py` works, with emphasis on **pairs analysis** and **reference-point selection**, and on cases where the current logic may not behave as expected.

## Overview

The script combines ascending and descending InSAR timeseries (LOS displacement) into **vertical** and **horizontal** displacement timeseries. It:

1. Loads two inputs (asc + desc .he5 or timeseries).
2. Optionally geocodes them to a common grid.
3. Pairs acquisition dates between the two tracks using a **shift schedule** (same-day or N-day offset).
4. Chooses **which track order** (asc, desc vs desc, asc) yields more pairs with a consistent temporal spacing (**mode count**).
5. Finds a **reference point** (lat/lon) that is valid in both tracks and applies it to both timeseries.
6. Decomposes each paired epoch into vertical and horizontal components and writes vert/horz .he5 and related outputs.

Entry point: `src/plotdata/cli/horzvert_timeseries.py`. Helpers: `plotdata.helper_functions` (`extract_window`, `find_reference_points_from_subsets`, etc.) and MintPy (`asc_desc2horz_vert`, `get_overlap_lalo`, objects, utils).

---

## Pairs analysis

### Purpose

Asc and desc have different acquisition calendars. We need to match dates so that each **pair** (date_1, date_2) corresponds to one epoch for the decomposition. The script builds pairs by **day shift**: for a given integer shift (in days), it matches date_1 from track 1 with date_2 = date_1 + shift from track 2. Only one pair per date in each track (greedy, first match wins).

### Repeat interval and shift schedule

- **Repeat interval** (`get_repeat_interval`): Inferred from metadata `mission` (or similar). Sentinel-1 → 12 days, TerraSAR-X → 11 days, CSK → 1 day; default 12 if unknown.
- **Step**: `step = ceil(repeat_interval / 2)` (e.g. 6 for S1).
- **`--intervals`** (`interval_index`, default 2): Number of “blocks” of shifts to consider. Each block is either a positive range (e.g. 0..6) or a negative range (e.g. -1..-6).

**Shift schedule** (`_shift_schedule` in `match_and_filter_pairs` and `match_and_filter_dates`):

- Builds a list of `(shift_days, block_index)` in a fixed order.
- For `interval_index = 2` and S1 (step=6):
  - Block 0 (positive): shifts 0, 1, …, 6.
  - Block 1 (negative): shifts -1, -2, …, -6.
- For larger `interval_index`, more blocks are added (further positive/negative ranges).
- **Order of processing**: shifts are processed in this schedule order. Pairs are assigned greedily: once a date from track 1 or track 2 is used, it is not reused. So same-day (shift 0) is tried first, then 1, -1, etc., and the first valid match wins.

### Matching algorithm (`match_dates`)

- Inputs: two date arrays (track 1 and track 2), and the schedule.
- For each `(shift_num, block_idx)` in the schedule:
  - Compute `shifted = track1_dates + shift_num` (days).
  - For each track1 date, if `shifted` equals a track2 date and neither date is already matched, form a pair and mark both as used.
- Output: array of pairs `(date_1, date_2)`, plus block counts, block_map, shift_map (per-pair shift in days).

So every pair satisfies: **date_2 = date_1 + shift** for some shift in the schedule. Delta (date_2 − date_1) in days is exactly that shift.

### Delta and “mode” (used for track order)

- For each pair, **delta_days = date_2 − date_1** (in days).
- **`delta_days_max_occurrence(delta_days_array)`**: Among **positive** deltas only, returns the **mode** (most frequent value) and its count. Negative deltas are ignored.
- So we get a single “most common positive spacing” and how many pairs have that spacing.

---

## How the pair analysis is used to choose “reference” (track order)

The script does **not** choose the geographic reference point from pairs; the user (or metadata) supplies **ref_lalo**. The pairs analysis is used to decide **which track is first (ts1) and which is second (ts2)** in the pipeline—i.e. the **ordering** of the two inputs.

### Swap logic in `main`

1. **Run pairing twice**:
   - `res_orig = run_fast(inps.file)`  → order (file1, file2), e.g. (desc, asc).
   - If not `--no-swap`: `res_swap = run_fast(reversed(inps.file))`  → order (file2, file1), e.g. (asc, desc).

2. **Choose ordering by mode count**:
   - `mode_val, mode_count = delta_days_max_occurrence(delta_fast)` for each run.
   - `chosen = res_swap if (not no_swap and res_swap["mode_count"] > res_orig["mode_count"]) else res_orig`.
   - So we **swap** to the reversed order only when the reversed order yields **strictly more** pairs at the most common positive delta. In a tie, the **original** order is kept.

3. **Effect**:
   - After this, `inps.file` and (if present) `inps.geom_file` may be reversed. All later steps (loading, reference point, matching again, decomposition) use this chosen order. So “reference” in the sense of “which track is ts1” is the one that, when placed first, gives **fewer** pairs at the positive mode; we use the order that **maximizes** the number of pairs at the mode. That maximizes the number of epochs we can decompose with a consistent temporal spacing.

4. **Important**: The **geographic** reference (ref_lalo) is unchanged; only the **role** of the two tracks (which is ts1 vs ts2) is chosen to maximize paired epochs. Because both orderings are analyzed, using only the count of pairs at the most common **positive** delta is sufficient: the same pairs appear in both runs, with opposite-sign deltas, so we simply pick the convention (which track is first) that yields the higher mode count.

---

## Reference point (geographic) selection

The **location** of the reference is given by `--ref-lalo` or by existing metadata (REF_LAT, REF_LON). The script then finds **which pixel** to use in each track so that both refer to the same ground point.

### Steps

1. **Extract window** (`extract_window`, in `helper_functions`):
   - For each track, take a small window of data and validity mask around the given ref_lalo (using metadata and coordinate conversion: geo → pixel indices). Window size is `--window-size` (default 3).
   - Returns for each track: a boolean mask of valid (non-NaN) pixels in that window, and the lat/lon of each pixel in the window.

2. **Find reference points from subsets** (`find_reference_points_from_subsets`):
   - With **two** subsets (asc and desc windows):
     - For each pixel in track 1’s window that is valid (mask True), get its (lat, lon).
     - Find the **nearest** pixel in track 2’s window (in lat/lon) and check if that pixel is also valid in track 2.
     - Among all such “both valid” candidates, choose the one **closest to the center** of the window (minimum Euclidean distance in pixel indices from the window center).
   - Returns one (lat, lon) that is used as the reference for **both** tracks; each track then converts that lat/lon to its own REF_Y, REF_X and subtracts that pixel’s value from the whole timeseries.

3. **Apply reference** (`process_reference_points`):
   - For each track, set metadata REF_LAT, REF_LON, REF_Y, REF_X and replace `ts.data` with `data - data[REF_Y, REF_X]` per slice.

So: **ref_lalo** fixes the approximate location; **find_reference_points_from_subsets** picks the **exact** ground point (and thus pixels) so that both tracks share the same reference and that point lies in a valid pixel in both grids.

---

## End-to-end flow (concise)

1. Parse options; expand `--period` into start/stop; apply `--exclude-dates`.
2. **Fast pairing** (dates only): load dates from both files, apply date limits, run `match_and_filter_pairs` with the shift schedule; compute mode and mode_count for (file1, file2) and, if not `--no-swap`, for (file2, file1); choose order by larger mode_count; write `image_pairs.txt`.
3. If `--dry-run`, stop.
4. Load both timeseries (with geometry, mask, optional geocoding); for each, extract window around ref_lalo and attach to object as `obj.window`.
5. **Reference points**: `find_reference_points_from_subsets([ts1.window, ts2.window], window_size)` → one ref (lat, lon); then `process_reference_points` to set REF_* and subtract reference from each track.
6. **Match again** on full timeseries: `match_and_filter_dates(ts1, ts2, inps)` (same shift schedule) → filter both timeseries to paired dates only; get delta_days, bperp, date_list, pairs.
7. **Overlap**: compute overlap in lat/lon; extract overlapping region from both; combine masks (AND).
8. **Decompose**: for each paired epoch, `asc_desc2horz_vert(...)` → vertical and horizontal; stack into vertical_timeseries and horizontal_timeseries.
9. Write vert/horz .he5, mask, and (if requested) timeseries outputs.

---

## Cases where the logic may not work

### Pairs and track order

1. **Tie in mode_count**  
   When both orderings give the same mode_count, the **original** argument order is kept. So the outcome depends on the order the user passes the two files. There is no further tie-breaker (e.g. orbit direction or total pairs).

2. **Wrong or missing repeat interval**  
   `get_repeat_interval` uses metadata `mission` (e.g. 'Sen' → 12, 'Csk' → 1). If mission is missing or wrong, the default is 12 days. For Cosmo-SkyMed (1-day) or TerraSAR-X (11-day), wrong repeat would yield wrong step and wrong shift blocks, so pairs could be too sparse or use inappropriate baselines.

3. **Different repeat intervals**  
   The code uses a single repeat interval for both tracks (from meta1, meta2). If asc and desc were from different missions with different repeat intervals, the single schedule would still be used; pairing could be suboptimal.

4. **Greedy matching order**  
   Pairs are assigned in schedule order (shift 0 first, then 1, -1, …). So “same-day” pairs are preferred. If the calendar is such that many dates could pair with either 0 or ±1 day, the result depends on this order. Changing the schedule order could change which dates get paired.

5. **Very uneven date counts**  
   If one track has many more dates than the other, most of the smaller track's dates may pair; the mode_count is still a count of pairs at the mode. The choice of order can still flip with small changes in the date lists (e.g. one extra date in one track). Analyzing both orderings makes the positive-delta criterion sufficient, so the earlier "only positive deltas" concern is addressed.

### Reference point

7. **No valid overlap in window**  
   If the user’s ref_lalo falls in a region that is **masked or NaN** in either track (e.g. layover, water, or outside coherence), there may be **no** pixel in the window that is valid in **both** tracks. Then `find_reference_points_from_subsets` raises: *"No valid reference points found in the selected window."*

8. **Geocoding / coordinate mismatch**  
   `extract_window` uses `utils.coordinate(metadata)` and checks bounds with X_FIRST/Y_FIRST when present. If one file is geocoded and the other is not yet, or if ref_lalo is in a different projection, the window or coordinate conversion could be wrong. The script assumes both inputs are on a consistent grid after any geocoding step.

9. **Window too large**  
   If the window is large relative to the grid, `extract_window` can reduce `window_size` automatically. The reference is still the closest valid point to the center in the (possibly reduced) window; in extreme cases the “reference” can drift from the user’s ref_lalo.

10. **Two-track overlap assumption**  
    `find_reference_points_from_subsets` is written to support one or two subsets. For two subsets it finds the best overlapping valid point. The logic assumes the two grids refer to the same area so that “nearest pixel in track 2” is meaningful; large misregistration could make the chosen point less intuitive.

### Date filtering and pairing

11. **`--start-date` / `--end-date` / `--period`**  
    These limit which dates participate in pairing. If they are different for the two tracks (e.g. one interval per track), the code still uses a single list of intervals and applies them to both; the logic in `limit_dates` and in the loader may not guarantee symmetric treatment if intervals are specified per file.

12. **`--exclude-dates`**  
    Excluded dates are removed before pairing. If many dates are excluded, the mode_count can drop and the chosen order can change, or too few pairs may remain.

13. **Repeated pairing**  
    Pairing is done twice: once in the “fast” path (dates only, to choose order and write image_pairs.txt) and again after loading full timeseries in `match_and_filter_dates`. The shift schedule is the same, but the date lists can differ if the first run uses `load_dates` (with limit_dates) and the second uses the already-loaded, geocoded objects. In principle they should match; any divergence (e.g. different date limits or rounding) could theoretically yield different pair sets.

### Other

14. **Metadata REF_LAT/REF_LON**  
    If the user does **not** pass `--ref-lalo`, ref is taken from the first file’s metadata. If that file was produced with a different grid or reference, ref_lalo might be unsuitable for the other track or for the current grid.

15. **HDFEOS vs timeseries**  
    The script supports both HDFEOS and MintPy timeseries types. Geometry and group names (e.g. bperp, mask) are handled with special cases. Unusual file layouts or renames could break assumptions.

---

## Summary

- **Pairs**: Built by matching track1 date with track2 date = date1 + shift, for shifts in a schedule derived from repeat interval and `--intervals`. Greedy by schedule order; each date used at most once.
- **Reference (track order)**: The script tries both (file1, file2) and (file2, file1), and picks the order that **maximizes the number of pairs** at the most common **positive** delta (mode_count). Geographic ref is still user/metadata ref_lalo.
- **Reference (location)**: ref_lalo is refined to a single (lat, lon) where both tracks have a valid pixel in a small window; that point is used to set REF_* and to subtract the reference from both timeseries.

The main failure modes are: wrong or missing repeat interval, no valid overlapping pixel at ref_lalo, tie in mode_count (order then depends on user order), and greedy/order-dependent pairing when multiple pairings are possible.
