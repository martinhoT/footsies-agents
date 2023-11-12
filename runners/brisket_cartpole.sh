#!/bin/sh
python3 main.py brisket --env CartPole-v1 --model-name brisket_cartpole -mN shallow_size 4 epsilon_decay_rate 0.001 --episodes 1000 --log-frequency 10000 --log-test-states-number 50000 "$@"