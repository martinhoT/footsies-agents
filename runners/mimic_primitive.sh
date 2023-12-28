#!/bin/sh
python3 main.py mimic \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    --model-name mimic_primitive \
    -mB over_primitive_actions true \
    "$@"