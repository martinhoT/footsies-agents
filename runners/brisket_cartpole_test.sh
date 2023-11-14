#!/bin/sh
python3 main.py brisket --env CartPole-v1 --model-name brisket_cartpole -mN shallow_size 4 q_value_min 0 q_value_max 500 epsilon 0 min_epsilon 0 --no-save --no-log -eS render_mode human "$@"