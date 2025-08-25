#!/bin/bash

# Define the container name
CONTAINER_NAME="minsarplotdata_container"

docker build -t minsarplotdata .

if [ -z "$1" ]; then
    echo "No scratch folder name provided. Using $SCRATCHDIR."
else
    echo "Using provided scratch folder name: $1"
    export SCRATCHDIR=$1
fi

docker run --name $CONTAINER_NAME --memory=24g -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e SCRATCHDIR=$SCRATCHDIR -v $SCRATCHDIR:$SCRATCHDIR -it minsarplotdata /bin/bash