#!/bin/sh
python3 main.py a2c \
    --torch \
    --env 'MountainCar-v0' \
    -mS \
        actor_hidden_layer_sizes_specification "4" \
        actor_hidden_layer_activation_specification ReLU \
        critic.environment "mountain car" \
    -mN \
        a2c.actor_entropy_loss_coef 0.2 \
        a2c.actor_optimizer.lr 0.001 \
        critic.discount 1.0 \
        critic.learning_rate 0.5 \
    -mB \
        use_q_table true \
        consider_opponent_action false \
        a2c.policy_cumulative_discount false \
        footsies false \
        use_simple_actions false \
    --log-frequency 1000 \
    --log-dir runs/a2c_mountain_car_qlearner \
    --model-name a2c_mountain_car_qlearner \
    "$@"