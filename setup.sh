#!/bin/bash
set -e

DATA_TARGET="/sc/projects/sci-aisc/pilotproject-automatic-protocols/data"

if [ -e "data" ]; then
    echo "data already exists — skipping symlink creation."
else
    ln -s "$DATA_TARGET" data
    echo "Created symlink: data -> $DATA_TARGET"
fi
