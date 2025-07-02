#!/bin/bash

# Define variables
IMAGE_NAME="mendeza/gradio_stream_demo"
IMAGE_TAG="0.0.1"
PLATFORM="linux/amd64"

# Ensure Docker Buildx is ready
docker buildx create --use --name multiarch || docker buildx use multiarch

# Build and push the image
docker buildx build --platform $PLATFORM -t $IMAGE_NAME:$IMAGE_TAG . --push
