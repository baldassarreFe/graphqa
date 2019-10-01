#!/usr/bin/zsh

EXTERNAL_USER=${EXTERNAL_USER-$(whoami)}
echo "Docker user  : $(whoami)@$(hostname)"
echo "External user: ${EXTERNAL_USER}"
echo "Working dir  : $(pwd)"

nvidia-smi

source /root/.zshrc
conda activate base
# Install as editable, otherwise pip tries to copy the whole data folder and it takes forever
pip install --editable --quiet /proteins/src && echo "Installed proteins@$(cd /proteins/src && git rev-parse --short HEAD)"

if [[ -n "$@" ]]; then
    echo -e "Executing CMD: $@\n"
    zsh -c "$@"
    EXIT_CODE=$?
else
    echo -e "Starting a zsh shell...\n"
    zsh
    EXIT_CODE=$?
fi

# Fix permissions for artifacts created as root
chown --from=$(id -u):$(id -g) --recursive --changes ${EXTERNAL_USER}:${EXTERNAL_USER} /proteins/runs

exit "$EXIT_CODE"