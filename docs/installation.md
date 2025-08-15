# Install package
```bash
pip install MinsarPlotData
pip install git+https://github.com/username/minsar.git
```
## Install `GDAL`
### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
```
### macOS(Homebrew)
```bash
brew install gdal
```
### Windows (conda recommended)
```bash
conda create -n geo_env python=3.12
conda activate geo_env
conda install -c conda-forge gdal geopandas
```
## Complete installation

```
pip install -r requirements.txt
```


### Step 1: Install Conda (if not installed)
Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Step 2: Create the environment from `environment.yml`
```bash
conda env create -f environment.yml
```

### Step 3: Activate the environment
```bash
conda activate geo_env
```

### Step 4: Update pip-only packages (optional):
```bash
conda activate geo_env
pip install -r requirements.txt
```

# Test installation

### Setup working directory (`SCRATCHDIR`)
Define env variable
```bash
export SCRATCHDIR=*Add/your/path*
```
Make directory
```bash
mkdir -p $SCRATCHDIR
```

## Download the data in the relative folders
```bash
BASE_DIR=$(pwd); \
for entry in \
"$SCRATCHDIR/MaunaLoaSenAT124/mintpy https://zenodo.org/records/16878080/files/S1_IW23_124_0059_0063_20150530_XXXXXXXX_N18623_N20314_W156162_W154265.he5?download=1" \
"$SCRATCHDIR/MaunaLoaSenDT87/mintpy https://zenodo.org/records/16877789/files/S1_IW12_087_0527_0531_20141116_XXXXXXXX_N18797_N20241_W156282_W154398.he5?download=1"; \
do folder=$(echo $entry | awk '{print $1}'); \
url=$(echo $entry | awk '{print $2}'); \
mkdir -p "$BASE_DIR/$folder" && cd "$BASE_DIR/$folder" && wget "$url" && cd "$BASE_DIR"; \
done
```

## Run code
```python
plotdata MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --template default  --period 20181001:20191031 --ref-lalo 19.50068 -155.55856 --resolution '01s' --contour 2 --lalo 19.461,-155.558 --num-vectors 40
```




