#!/bin/bash

set -euo pipefail

# Run the Merlin grid inside an AWS GPU-enabled Docker container.
#
# Typical usage on the remote Linux server from the repo root:
#   docker images -f "dangling=true" -q --no-trunc | xargs -r docker rmi && \
#   git pull && \
#   GPU_OPTION="--runtime=nvidia" ./run_merlin_container_aws.sh
#
# Or with the default modern Docker GPU flag:
#   GPU_OPTION="--gpus all" ./run_merlin_container_aws.sh
#
# To only start the container (and NOT run the grid script),
# you can call:
#   ./run_merlin_container_aws.sh container-only

IMAGE_NAME="${IMAGE_NAME:-generative-recommenders-merlin}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-generative-recommenders-merlin}"

# If USE_CPU_ONLY=1 is set in the environment, do not request GPUs
# and hide CUDA devices inside the container so training runs on CPU.
USE_CPU_ONLY="${USE_CPU_ONLY:-0}"

# GPU_OPTION is expected to be passed in from the environment.
# Fall back to a reasonable default if not provided, unless CPU-only.
if [ "${USE_CPU_ONLY}" = "1" ]; then
  GPU_OPTION=""
else
  GPU_OPTION="${GPU_OPTION:---gpus all}"
fi

EXTRA_ENV_ARGS=()
if [ "${USE_CPU_ONLY}" = "1" ]; then
  # Hide GPUs from PyTorch so torch.cuda.is_available() is False.
  EXTRA_ENV_ARGS+=("-e" "CUDA_VISIBLE_DEVICES=")
fi

# Host directories to persist artifacts and data. Defaults assume the script
# is run from the repo root on the host.
HOST_ROOT="${HOST_ROOT:-$(pwd)}"
HOST_LOGS_DIR="${HOST_LOGS_DIR:-${HOST_ROOT}/logs}"
HOST_CKPTS_DIR="${HOST_CKPTS_DIR:-${HOST_ROOT}/ckpts}"
HOST_EXPS_DIR="${HOST_EXPS_DIR:-${HOST_ROOT}/exps}"
HOST_MERLIN_TMP_DIR="${HOST_MERLIN_TMP_DIR:-${HOST_ROOT}/tmp/merlin}"

mkdir -p "${HOST_LOGS_DIR}" "${HOST_CKPTS_DIR}" "${HOST_EXPS_DIR}" "${HOST_MERLIN_TMP_DIR}"

# Parse command line arguments
RUN_MODE="full"
if [ "${1:-}" = "container-only" ]; then
  RUN_MODE="container-only"
fi

echo "Building Docker image using build_image_aws.sh"
bash ./build_image_aws.sh

# Remove any existing container with the same name
if docker ps -a -q -f "name=${CONTAINER_NAME}" >/dev/null 2>&1 && \
   [ -n "$(docker ps -a -q -f "name=${CONTAINER_NAME}")" ]; then
  echo "Container ${CONTAINER_NAME} already exists. Removing it..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Running Docker container ${CONTAINER_NAME} in background mode"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart always \
  ${GPU_OPTION} \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "${HOST_LOGS_DIR}:/workspace/logs" \
  -v "${HOST_CKPTS_DIR}:/workspace/ckpts" \
  -v "${HOST_EXPS_DIR}:/workspace/exps" \
  -v "${HOST_MERLIN_TMP_DIR}:/workspace/tmp/merlin" \
  "${IMAGE_NAME}:${IMAGE_TAG}"

echo "Container ${CONTAINER_NAME} started successfully!"

if [ "${RUN_MODE}" = "container-only" ]; then
  echo "Running in container-only mode. Container is ready but run_merlin_grid.py will not execute."
  echo "You can exec into the container with:"
  echo "  docker exec -it ${CONTAINER_NAME} bash"
  exit 0
fi

sleep 2

echo "Executing run_merlin_grid.py inside container ${CONTAINER_NAME}"
docker exec -it "${CONTAINER_NAME}" python3 run_merlin_grid.py

echo "run_merlin_grid.py finished."
