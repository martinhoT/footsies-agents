#!/bin/sh
python3 main.py game_model \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 100000 \
    -eB by_example true \
    -mN \
        opponent_action_dim 10 \
        move_transition_scale 10 \
        optimize_frequency 100000 \
    -mB \
        by_primitive_actions false \
    -mS \
        hidden_layer_sizes_specification 128,128,128 \
    --log-dir runs/game_model_adjusted \
    --model-name game_model_adjusted \
    "$@"