#!/bin/sh
python3 main.py brisket \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --footsies-wrapper-fs \
    --wrapper-time-limit 5940 \
    --penalize-truncation -1.0 \
    --model-name brisket_frameskipped_self-play \
    "$@"