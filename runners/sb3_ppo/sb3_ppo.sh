#!/bin/sh
# 300k timesteps should amount to 1 min
python3 main.py sb3.ppo \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --time-steps 10000000 \
    --model-name ppo \
    -mS \
        policy MlpPolicy \
    -mN \
        gamma 1.0 \
        ent_coef 0.01 \
    "$@"