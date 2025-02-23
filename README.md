# PlotData
Scripts to plot InSAR, GPS and Seismcity data. Need a working python installation including `MintPy`.


# Installation
- Set environment variables:
```
export SCRATCHDIR=~/Downloads/test
export PLOTDATA_HOME=~/tools/PlotData
export GPSDIR=~/Downloads/GPSdata
```
- Prepend to your $PATH:
```
export PATH=${PLOTDATA_HOME}/src/plotdata/cli:$PATH
export PYTHONPATH=${PLOTDATA_HOME}/src:$PYTHONPATH
```
- Get the InSAR and GPS data for Hawaii:
```
mkdir -p $SCRATCHDIR
cd $SCRATCHDIR
wget http://149.165.154.65/data/HDF5EOS/MaunaLoa/MaunaLoaSen.tar
tar xvf MaunaLoaSen.tar

mkdir -p $GPSDIR
cd $GPSDIR
wget http://149.165.154.65/data/HDF5EOS/MaunaLoa/GPSdata.tar
tar xvf GPSdata.tar 
```
- Clone code to your tools directory:
```
mkdir -p $PLOTDATA_HOME
cd $PLOTDATA_HOME/..
git clone https://github.com/geodesymiami/PlotData.git
```
- run testdata:
```
cd $SCRATCHDIR
plot_data.py --help
plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --period 20221127-20221219 --plot-type velocity --ref-point 19.55,-155.45
```
Run the Notebook `run_MaunaLoa.ipynb` to create all relevant data. It calls `plot_data.py`.
The data will be located in `ls $SCRATCHDIR/MaunaLoa/*/*`.  

You also can run in Jupyer Lab `plot_data.ipynb` which is the original ipython version. To change plot options, adjust `cmd` line below the `main` function as needed.  
`--save_gbis` saves data in GBIS format.

## ViewPS for MiaplPy results in radar coordinates

The package contains `viewPS.py` for displaying MiaplPy results 

To test, download example data (28GB) (`$SCRATCHDIR` is needed, e.g. `export SCRATCHDIR=~/Downloads`)
```
rsyncFJ MiamiSenAT48/miaplpy_MDCBeach_201601_202310/network_delaunay_4
rsyncFJ MiamiSenAT48/DEM
rsyncFJ MiamiSenAT48/miaplpy_MDCBeach_201601_202310/inputs

cd /Users/famelung/Downloads/scratch/MiamiSenAT48/miaplpy_MDCBeach_201601_202310/network_delaunay_4/

viewPS.py S1*PS.he5 --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 &
viewPS.py S1_IW23_048_0081_0083_20160412_20231021_N25765_N25980_W080147_W080115_Del4PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --vlim -3 3 --ref-lalo 25.87609,-80.12213 --dem ../../DEM/MiamiBeach.tif --dem-noshade &
```
For the full suite of examples clone the PlotData-notebooks repo (into `rsmas_insar/notebooks` so that the `cdpdn` alias works) and run
https://github.com/geodesymiami/PlotData-notebooks/blob/main/run_Miami.ipynb

