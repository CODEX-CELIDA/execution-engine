#!/bin/bash

CONTAINER_NAME="postgres-pytest-celida"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="mysecretpassword"
POSTGRES_PORT="5434"

# Check if the container already exists
if [ "$(docker ps -a -f name=^/${CONTAINER_NAME}$ --format '{{.Names}}')" == $CONTAINER_NAME ]; then
    echo "Container already exists"

    # Check if the container is not running
    if [ "$(docker ps -f name=^/${CONTAINER_NAME}$ --format '{{.Names}}')" != $CONTAINER_NAME ]; then
        echo "Starting existing container"
        docker start $CONTAINER_NAME
    else
        echo "Container is already running"
    fi
else
    echo "Creating and starting a new container"
    docker run --name $CONTAINER_NAME -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD -e POSTGRES_USER=$POSTGRES_USER -p $POSTGRES_PORT:5432 -d postgres --restart=always
fi
