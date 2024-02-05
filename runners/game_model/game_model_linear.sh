#!/bin/sh
python3 main.py game_model \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mN \
        move_transition_scale 1 \
        optimize_frequency 1000 \
        learning_rate 0.01 \
    -mB \
        by_primitive_actions false \
    -mS \
        hidden_layer_sizes_specification "" \
        hidden_layer_activation_specification Identity \
    --log-frequency 100000 \
    --log-dir runs/game_model_linear \
    --model-name game_model_linear \
    "$@"