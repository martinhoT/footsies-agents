#!/bin/sh
python3 main.py game_model \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mN \
        move_transition_scale 10 \
        mini_batch_size 100000 \
        learning_rate 0.1 \
    -mB \
        by_primitive_actions false \
    -mS \
        hidden_layer_sizes_specification 128 \
        hidden_layer_activation_specification LeakyReLU \
    --log-frequency 100000 \
    --log-dir runs/game_model_one_layer \
    --model-name game_model_one_layer \
    "$@"