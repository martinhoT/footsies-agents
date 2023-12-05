#!/bin/sh
# 350k timesteps should amount to ~30 mins
python3 main.py sb3.ppo \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --time-steps 350000 \
    -mS \
        policy MlpPolicy \
    -mN \
        gamma 1.0 \
        ent_coef 0.01 \
    "$@"