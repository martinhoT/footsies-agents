#!/bin/sh
python3 main.py a2c \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -mS \
        actor_hidden_layer_sizes_specification "64,64" \
        actor_hidden_layer_activation_specification ReLU \
        actor_hidden_layer_sizes_specification "128,128" \
        actor_hidden_layer_activation_specification ReLU \
    -mN \
        a2c.actor_entropy_loss_coef 0.1 \
        a2c.actor_optimizer.lr 0.001 \
        critic.discount 1.0 \
        critic.learning_rate 0.5 \
    -mB \
        use_q_table true \
        consider_opponent_action true \
        a2c.policy_cumulative_discount false \
    --log-frequency 1000 \
    --log-dir runs/a2c_qlearner \
    --model-name a2c_qlearner \
    "$@"