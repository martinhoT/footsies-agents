#!/bin/sh
python3 main.py sb3.SAC \
    --torch \
    --env 'MountainCarContinuous-v0' \
    --time-steps 1000 \
    --diayn \
    --diayn-skill-dim 20 \
    --diayn-no-baseline \
    --diayn-discriminator-learning-rate 0.001 \
    --diayn-discriminator-hidden-layer-sizes-specification "64,64" \
    --diayn-discriminator-hidden-layer-activation-specification ReLU \
    --model-name a2c_mountain_car_diayn_sac \
    --log-frequency 1000 \
    --log-dir runs/a2c_mountain_car_diayn_sac \
    "$@"
