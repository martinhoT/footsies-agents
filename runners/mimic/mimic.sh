#!/bin/sh
python3 main.py mimic \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mN \
        move_transition_scale 10 \
        optimize_frequency 100000 \
        learning_rate 0.01 \
    -mB \
        by_primitive_actions false \
    -mS \
        hidden_layer_sizes_specification "64" \
        hidden_layer_activation_specification LeakyReLU \
    --log-frequency 100000 \
    --log-dir runs/mimic \
    --model-name mimic \
    "$@"