#!/usr/bin/env bash

cp -r ../../torchgraphs torchgraphs
cp ../conda.yaml conda.yaml

docker build -t proteins:v0 .

rm conda.yaml
rm -r torchgraphs
