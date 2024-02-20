#!/bin/sh
python3 main.py the_one \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -mN \
        representation_dim 32 \
        reaction_time_emulator.inaction_probability 0.0 \
        reaction_time_emulator_minimum 15.0 \
        reaction_time_emulator_maximum 29.0 \
    -mB \
        consider_actions_in_representation false \
    -mS \
        representation_hidden_layer_sizes_specification "128,128" \
        representation_hidden_layer_activation_specification ReLU \
    --log-frequency 10000 \
    --log-dir runs/the_one \
    --model-name the_one \
    "$@"