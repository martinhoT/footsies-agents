#!/bin/sh
python3 main.py mimic \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mN \
        mini_batch_size 1 \
        learning_rate 0.01 \
        move_transition_scale 30 \
    -mB \
        frameskipping true \
        by_primitive_actions false \
    -mS \
        hidden_layer_sizes_specification "" \
        hidden_layer_activation_specification Identity \
    --log-frequency 10000 \
    --log-dir runs/mimic_linear_frameskip \
    --model-name mimic_linear_frameskip \
    "$@"