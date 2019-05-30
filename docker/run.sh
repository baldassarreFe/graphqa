#!/usr/bin/env bash

ENTRYPOINT_PATH="${ENTRYPOINT_PATH-$(realpath ./entrypoint.sh)}"
SOURCE_PATH="${SOURCE_PATH-$(realpath ..)}"
DATA_PATH="${DATA_PATH-$(realpath ~/experiments/proteins/data)}"
RUNS_PATH="${RUNS_PATH-$(realpath ~/experiments/proteins/runs)}"

docker run --runtime=nvidia --ipc=host --rm -it \
    --workdir='/root/proteins' \
    --env="EXTERNAL_USER=$(id -u)" \
    --mount "type=bind,source=${SOURCE_PATH},target=/root/proteins" \
    --mount "type=bind,source=${DATA_PATH},target=/root/experiments/proteins/data,readonly" \
    --mount "type=bind,source=${RUNS_PATH},target=/root/experiments/proteins/runs" \
    --mount="type=bind,source=${ENTRYPOINT_PATH},target=/entrypoint.sh" \
    --entrypoint=/entrypoint.sh \
    proteins:v0 \
    "$@"