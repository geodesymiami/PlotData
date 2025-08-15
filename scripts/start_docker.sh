#!/bin/bash
docker build -t minsarplotdata .

if [ -z "$1" ]; then
    echo "No scratch folder name provided. Using default name."
else
    echo "Using provided scratch folder name: $1"
    export SCRATCHDIR=$1
fi

docker run -e SCRATCHDIR=$SCRATCHDIR -v $SCRATCHDIR:$SCRATCHDIR -it minsarplotdata