#!/bin/sh
python3 main.py the_one_vanilla \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 1000 \
    --footsies-self-play \
    --footsies-self-play-max-snapshots 10 \
    --footsies-self-play-snapshot-interval 2000 \
    --footsies-self-play-switch-interval 100 \
    --footsies-self-play-mix-bot 1 \
    --wrapper-time-limit 1000 \
    --penalize-truncation -1 \
    -eS \
        sync_mode synced_non_blocking \
    "$@"