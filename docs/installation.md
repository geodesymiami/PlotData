# Install package
```bash
pip install MinsarPlotData
```
Go to repo folder
```bash
cd PlotData #Temporary name
```
You can use one of the following options:
- Use [Docker](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md#instal-with-docker)
- Follow the [Regular installation](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md#regular-installation)

---
# Instal with Docker
## Simplified approach
Pass an argument to the following `CMD` if:
- You don't have a `SCRATCHDIR` directory set up and you don't want to use the default (Built on `$HOME/scratchdir`):
```bash
bash PlotData/scripts/start_docker.sh path/to/scratch
```
**Otherwise** just run:
```bash
bash PlotData/scripts/start_and_build_docker.sh
```
## Manual approach
Build **Docker image**.
```bash
docker build -t minsarplotdata .
```
If you don't have a `SCRATCHDIR` directory set up and you don't want to use the default (Built on `$HOME/scratchdir`), run:
```bash
export SCRATCHDIR=/path/to/scratch
```
Run container
```bash
CONTAINER_NAME="minsarplotdata_container"
docker run --name $CONTAINER_NAME --memory=4g -e SCRATCHDIR=$SCRATCHDIR -v $SCRATCHDIR:$SCRATCHDIR -it minsarplotdata
```
---
# Regular Installation
## Option 1: Install with Conda

### Step 1: Install Conda (if not installed)
Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Step 2: Create the environment from `environment.yml and install dependencies`
```bash
conda env create -f environment.yml
```

## Option 2: Install with Pip (if you already have Python + GDAL)

### Step 1: Make sure GDAL is installed system-wide
- **On Linux/macOS:**
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
```

- **macOS (Homebrew):**
```bash
brew install gdal
```

-	On Windows: use [Conda](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md#option-1-install-with-conda) instead.

### Step 2: Create and activate a virtual environment (recommended)
```bash
python3 -m venv plotdata
source plotdata/bin/activate   # On Linux/macOS
```

### Step 3: Install Python dependencies with pip
```bash
pip install -r requirements.txt
```

# Test installation

### Setup working directory (`SCRATCHDIR`)
Define env variable (if you used [Docker](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md#instal-with-docker) you can skip this part):
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















