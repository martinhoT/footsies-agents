#!/bin/sh
python3 main.py the_one \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -mN \
        representation_dim 64 \
        reaction_time_emulator.inaction_probability 0.0 \
        reaction_time_emulator.multiplier 0.0 \
        reaction_time_emulator_minimum 0.0 \
        a2c.actor_entropy_loss_coef 0.5 \
    -mB \
        consider_actions_in_representation false \
        opponent_model_learn false \
        actor_critic_frameskip true \
        opponent_model_frameskip true \
    -mS \
        representation_hidden_layer_sizes_specification "256" \
        representation_hidden_layer_activation_specification ReLU \
    --log-frequency 10000 \
    --log-dir runs/the_one_no_react_ent_multi \
    --model-name the_one_no_react_ent_multi \
    --no-save \
    --no-log \
    -eS \
        render_mode human \
        sync_mode synced_non_blocking \
    -eB \
        fast_forward false \
        vs_player true \
    "$@"