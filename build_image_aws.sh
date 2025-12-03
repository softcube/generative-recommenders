#!/bin/bash

set -euo pipefail

# Simple helper to build the AWS CUDA training image for this repo.
# Uses Dockerfile_aws in the project root.
#
# Usage:
#   bash ./build_image_aws.sh
#
# Optionally override the image name/tag:
#   IMAGE_NAME=generative-recommenders-merlin IMAGE_TAG=mytag bash ./build_image_aws.sh

IMAGE_NAME="${IMAGE_NAME:-generative-recommenders-merlin}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG} using Dockerfile_aws"

docker build \
  -f Dockerfile_aws \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

echo "Done: ${IMAGE_NAME}:${IMAGE_TAG}"

