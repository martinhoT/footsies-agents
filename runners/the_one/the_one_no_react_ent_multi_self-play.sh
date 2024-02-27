#!/bin/sh
python3 main.py the_one \
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
    --log-dir runs/the_one_no_react_ent_multi_self-play \
    --model-name the_one_no_react_ent_multi_self-play \
    --hogwild \
    --hogwild-cpus 12 \
    --hogwild-n-workers 6 \
    --footsies-self-play \
    --footsies-self-play-snapshot-freq 1000 \
    --footsies-self-play-max-snapshots 100 \
    --footsies-self-play-mix-bot 50 \
    --wrapper-time-limit 1000 \
    --penalize-truncation -1 \
    --episodes 1000 \
    -eS \
        sync_mode synced_non_blocking \
    "$@"