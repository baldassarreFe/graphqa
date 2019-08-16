#!/usr/bin/env bash

cp ../conda.yaml conda.yaml

docker build -t baldassarrefe/proteins:v0 .

rm conda.yaml
