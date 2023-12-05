#!/bin/sh
python3 main.py brisket \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --footsies-wrapper-fs \
    --footsies-self-play \
    --footsies-self-play-snapshot-freq 1000 \
    --footsies-self-play-max-snapshots 100 \
    --footsies-self-play-mix-bot 50 \
    --footsies-self-play-port 11001 \
    --wrapper-time-limit 90 \
    --penalize-truncation -1.0 \
    --model-name brisket_frameskipped_self-play \
    "$@"