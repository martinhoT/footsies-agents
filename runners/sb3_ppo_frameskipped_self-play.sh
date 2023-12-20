#!/bin/sh
# 300k timesteps should amount to 1 min
python3 main.py sb3.ppo \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --footsies-wrapper-fs \
    --footsies-self-play \
    --footsies-self-play-snapshot-freq 1000 \
    --footsies-self-play-max-snapshots 100 \
    --footsies-self-play-mix-bot 50 \
    --footsies-self-play-port 11001 \
    --time-steps 10000000 \
    --model-name ppo_frameskipped_self-play \
    -mS \
        policy MlpPolicy \
    -mN \
        gamma 1.0 \
        ent_coef 0.01 \
    "$@"