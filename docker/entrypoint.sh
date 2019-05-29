#!/usr/bin/env bash

EXTERNAL_USER=${EXTERNAL_USER-$(whoami)}
echo "Docker user: $(whoami)@$(hostname)"
echo "External user: ${EXTERNAL_USER}"

source /root/.zshrc
conda activate base
pip install /proteins > /dev/null && echo "Installed proteins@$(cd /proteins && git rev-parse --short HEAD)"

if [[ -n "$@" ]]; then
    echo -e "Executing CMD: $@\n"
    zsh -c "$@"
else
    echo -e "Starting a zsh shell...\n"
    zsh
fi

# Fix permissions for artifacts created as root
chown --from=$(id -u):$(id -g) --recursive --changes ${EXTERNAL_USER}:${EXTERNAL_USER} /runs