#!/bin/sh
python3 main.py a2c \
    --torch \
    --env 'MountainCar-v0' \
    --diayn \
    --diayn-skill-dim 20 \
    --diayn-no-baseline \
    --diayn-discriminator-learning-rate 0.001 \
    --diayn-discriminator-hidden-layer-sizes-specification "64,64" \
    --diayn-discriminator-hidden-layer-activation-specification ReLU \
    -mS \
        actor_hidden_layer_sizes_specification "32" \
        actor_hidden_layer_activation_specification ReLU \
        critic_hidden_layer_sizes_specification "32" \
        critic_hidden_layer_activation_specification ReLU \
    -mN \
        a2c.discount 0.99 \
        a2c.actor_lambda 0.0 \
        a2c.critic_lambda 0.0 \
        a2c.actor_entropy_loss_coef 0.1 \
        a2c.actor_optimizer.lr 0.001 \
        a2c.critic_optimizer.lr 0.001 \
    --model-name a2c_mountain_car_diayn \
    --log-frequency 1000 \
    --log-dir runs/a2c_mountain_car_diayn \
    "$@"
