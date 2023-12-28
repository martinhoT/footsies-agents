#!/bin/sh
python3 main.py autoencoder \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mB \
        encoded_dim 3 \
        normalized false \
        include_sequentiality_loss false \
    "$@"