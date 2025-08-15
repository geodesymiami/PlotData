#!/bin/bash

# Default SCRATCHDIR path on the host machine
DEFAULT_SCRATCHDIR="$HOME/scratchdir"

# Check if SCRATCHDIR is already set on the host
if [ -z "$SCRATCHDIR" ]; then
    echo "SCRATCHDIR is not set. Using default: $DEFAULT_SCRATCHDIR"
    SCRATCHDIR="$DEFAULT_SCRATCHDIR"
fi

# Check if SCRATCHDIR is a symbolic link
if [ -L "$SCRATCHDIR" ]; then
    echo "SCRATCHDIR is a symbolic link. Resolving to its target."
    SCRATCHDIR=$(readlink -f "$SCRATCHDIR")
fi

# Check if the directory exists on the host
if [ ! -d "$SCRATCHDIR" ]; then
    echo "SCRATCHDIR does not exist on the host. Creating it at $SCRATCHDIR."
    mkdir -p "$SCRATCHDIR"
    chmod 777 "$SCRATCHDIR"
else
    echo "SCRATCHDIR already exists on the host at $SCRATCHDIR."
fi

# Export SCRATCHDIR for the container
export SCRATCHDIR
echo "SCRATCHDIR=$SCRATCHDIR" > /tmp/scratchdir.env