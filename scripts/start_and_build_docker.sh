#!/bin/bash

# Define the container name
CONTAINER_NAME="minsarplotdata_container"

# Build the Docker image
# TODO remove --no-cache after debugging
docker build --no-cache -t minsarplotdata .

# Check if a scratch folder name is provided
if [ -z "$1" ]; then
    echo "No scratch folder name provided. Using default: $SCRATCHDIR."
else
    echo "Using provided scratch folder name: $1"
    export SCRATCHDIR=$1
fi

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "DISPLAY is not set. Attempting to set it automatically..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Use XQuartz
        export DISPLAY=:0
        echo "Set DISPLAY to :0 for macOS. Ensure XQuartz is running and 'xhost + 127.0.0.1' is executed."
    else
        # Linux: Use the host's DISPLAY
        export DISPLAY=$DISPLAY
        echo "Set DISPLAY to $DISPLAY for Linux."
    fi
fi

# Check if X server is accessible
if ! xhost >&/dev/null; then
    echo "X server is not accessible. Ensure X server is running and 'xhost +local:docker' is executed."
    exit 1
fi

# Run the Docker container
docker run --rm --name $CONTAINER_NAME \
    --memory=24g \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e SCRATCHDIR=$SCRATCHDIR \
    -v $SCRATCHDIR:$SCRATCHDIR \
    -e MPLBACKEND=Agg \
    -it minsarplotdata xvfb-run -s "-screen 0 1920x1080x24" /bin/bash