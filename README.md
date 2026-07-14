[![CircleCI](https://dl.circleci.com/status-badge/img/gh/geodesymiami/PlotData/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/geodesymiami/PlotData/tree/main)
# PlotData
## TODOs
- [ ] Change examples
- [ ] Create notebook tutorial
- [ ] Run tests
# 1. [Installation](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md)

# 2. Download test data

Use the `get_plotdata_testdata.py` script to download sample data (--quick for hvGalapagos_mintpy.tar.gz) 
```bash
get_plotdata_testdata.py --quick
```

# 3. Run code
The package is coupled to **MintPy** and **MiaplPy**, so it leverages their data structures.

## Example Usage
Once you have test data (see section 2), you can run PlotData:

```python
plotdata hvGalapagos/mintpy --template default --ref-lalo -0.81 -91.190 --lalo -0.82528 -91.13791

# Example with Mauna Loa data (requires additional download)
plotdata MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --template default  --period 20181001:20191031 --ref-lalo 19.50068 -155.55856 --resolution '01s' --contour 2 --lalo 19.461,-155.558 --num-vectors 40
```

Check the [Installation guide](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md) for more details.

## Fault transect plotting (plot_fault_transect.py)

`plot_fault_transect.py` takes a fault trace KMZ and 1-4 HDFEOS5 timeseries inputs
(S1_*.he5 files or mintpy/miaplpy directories) and plots the across-fault
displacement offset sampled along the fault (`--plot-type map`), fault-perpendicular
profiles (`--plot-type profile`), and/or stacked across-fault displacement
timeseries (`--plot-type timeseries`). Use `--plot-type all` (default) for all three.
Profile and timeseries locations use `--plot-step-factor N` (every Nth `--along-step`
sampling point; default 2). Vertical spacing in stacked profile/timeseries plots:
`--stack-offset` (cm; default auto). Layouts for profiles: separate, subplot, stacked.
All plotted data is exported to companion `.txt` files and an `index.html` is
generated. Code lives in `src/plotdata/fault_transect/`; matplotlib is isolated
in `fault_transect/backends/matplotlib_backend.py`.

```bash
# Multi-segment KMZ: joint fault is created automatically when needed
plot_fault_transect.py PFS_Pernicana_faults_system_.kmz EtnaSenA44/mintpy --fault-segment 1,2,4,5,6,7,8,9,10,11 --no-display

# Map + stacked profiles for one dataset
plot_fault_transect.py PFS_Pernicana_faults_system_joint.kmz EtnaSenA44/mintpy --tag Pernicana --no-display

# Stacked timeseries with period boundary markers
plot_fault_transect.py FiandacaFault_FA.kmz EtnaSenA44/mintpy --plot-type timeseries --period 20141001:20181222,20181228:20260701 --no-display

# Two periods side-by-side, 10 profiles in a 10x1 subplot
plot_fault_transect.py fault_joint.kmz EtnaSenA44/mintpy --plot-type profile --plot-layout subplot --profile-count 10 --period 20141020:20181231,20190101:20260626 --no-display
```

Default output directory: `<project>/transects_mintpy/` (or `transects_miaplpy/`).
Run unit tests with `python -m unittest discover -s src/plotdata/fault_transect/tests`.
