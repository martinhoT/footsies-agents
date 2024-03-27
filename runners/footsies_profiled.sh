#!/bin/sh
agent=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

# This is supposed to be the very last script in the runner script chain
python3 -m torch.utils.bottleneck main.py $agent \
    --env.torch \
    --env.footsies-wrapper-norm \
    --env.footsies-wrapper-acd \
    --env.wrapper-time-limit 3000 \
    --env.kwargs \
        game_path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --env.kwargs \
        dense_reward False \
    --env.kwargs \
        render_mode human \
    --misc.log-frequency 1000 \
    "$@"