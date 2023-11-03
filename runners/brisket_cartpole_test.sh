#!/bin/sh
python3 main.py brisket --env CartPole-v1 --model-name brisket_cartpole -mF epsilon 0 min_epsilon 0 --episodes 1000 --no-save "$@"