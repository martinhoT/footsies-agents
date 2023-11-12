#!/bin/sh
python3 main.py brisket --env CartPole-v1 --model-name brisket_cartpole -mN shallow_size 4 epsilon 0 min_epsilon 0 --no-save --no-log -eS render_mode human "$@"