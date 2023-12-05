#!/bin/sh
python3 main.py brisket \
    --env CartPole-v1 \
    --model-name brisket_cartpole \
    -mN \
        shallow_size 4 \
        epsilon_decay_rate 0.0002 \
        learning_rate 0.001 \
        discount_factor 1.0 \
        alpha 0.9 \
        q_value_min 0 \
        q_value_max 500 \
    --episodes 100000 \
    --log-frequency 100000 \
    --log-test-states-number 50000 \
    "$@"