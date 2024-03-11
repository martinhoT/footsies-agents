#!/bin/sh
python3 main.py the_one_vanilla \
    --name the_one_vanilla_multi \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 1000 \
    --hogwild \
    --hogwild-cpus 12 \
    --hogwild-n-workers 6 \
    "$@"