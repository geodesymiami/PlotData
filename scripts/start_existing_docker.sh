#!/bin/bash

# Define the container name
CONTAINER_NAME="minsarplotdata_container"

# Check if the container already exists
EXISTING_CONTAINER=$(docker ps -aq -f name=$CONTAINER_NAME)

if [ -n "$EXISTING_CONTAINER" ]; then
    # If the container exists, check if it's running
    RUNNING_CONTAINER=$(docker ps -q -f name=$CONTAINER_NAME)
    if [ -n "$RUNNING_CONTAINER" ]; then
        echo "Container is already running. Attaching to it..."
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        echo "Starting existing container..."
        docker start $CONTAINER_NAME
        docker exec -it $CONTAINER_NAME /bin/bash
    fi
else
    echo "No existing container found for minsarplotdata. Please run build_and_start_docker.sh first."
fi