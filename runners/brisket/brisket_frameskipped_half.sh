#!/bin/sh
python3 main.py brisket \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --footsies-wrapper-fs \
    --model-name brisket_frameskipped_half \
    -mN shallow_size 16 \
    "$@"