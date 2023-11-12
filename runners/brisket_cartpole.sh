#!/bin/sh
python3 main.py brisket --env CartPole-v1 --model-name brisket_cartpole -mN shallow_size 4 --log-frequency 50000 --log-test-states-number 50000 "$@"