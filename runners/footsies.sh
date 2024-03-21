#!/bin/sh
agent=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

# This is supposed to be the very last script in the runner script chain
python3 main.py $agent \
    --env.torch \
    --env.footsies-wrapper-norm \
    --env.footsies-wrapper-acd \
    --env.wrapper-time-limit 3000 \
    --env.kwargs \
        footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
        dense_reward false \
        render_mode human \
    --misc.log-frequency 1000 \
    "$@"
