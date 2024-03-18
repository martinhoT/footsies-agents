#!/bin/sh
agent=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

# This is supposed to be the very last script in the runner script chain
python3 main.py $agent \
    --torch \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    --log-frequency 1000 \
    --wrapper-time-limit 1000 \
    -eB \
        dense_reward false \
    -eS \
        render_mode human \
    "$@"
