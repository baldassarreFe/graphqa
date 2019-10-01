#!/usr/bin/env bash

ENTRYPOINT_PATH="${ENTRYPOINT_PATH-$(realpath ./entrypoint.sh)}"
SOURCE_PATH="${SOURCE_PATH-$(realpath ..)}"
DATA_PATH="${DATA_PATH-$(realpath ../data)}"
RUNS_PATH="${RUNS_PATH-$(realpath ../runs)}"

docker run --runtime=nvidia --ipc=host --rm -it \
    --workdir='/proteins' \
    --env="EXTERNAL_USER=$(id -u)" \
    --mount "type=bind,source=${SOURCE_PATH},target=/proteins" \
    --mount "type=bind,source=${DATA_PATH},target=/proteins/data,readonly" \
    --mount "type=bind,source=${RUNS_PATH},target=/proteins/runs" \
    --mount="type=bind,source=${ENTRYPOINT_PATH},target=/entrypoint.sh" \
    --entrypoint=/entrypoint.sh \
    baldassarrefe/proteins:v0 \
    "$@"