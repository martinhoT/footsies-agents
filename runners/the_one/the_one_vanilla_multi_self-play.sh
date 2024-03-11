#!/bin/sh
python3 main.py the_one_vanilla \
    --name the_one_vanilla_multi_self-play \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 1000 \
    --hogwild \
    --hogwild-cpus 12 \
    --hogwild-n-workers 6 \
    --footsies-self-play \
    --footsies-self-play-snapshot-freq 1000 \
    --footsies-self-play-max-snapshots 100 \
    --footsies-self-play-mix-bot 50 \
    --wrapper-time-limit 1000 \
    --penalize-truncation -1 \
    --episodes 1000 \
    -eS \
        sync_mode synced_non_blocking \
    "$@"