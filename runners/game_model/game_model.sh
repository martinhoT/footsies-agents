#!/bin/sh
python3 main.py game_model \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 100000 \
    -eB by_example true \
    -mN \
        opponent_action_dim 10 \
    -mB \
        by_primitive_actions false \
    "$@"