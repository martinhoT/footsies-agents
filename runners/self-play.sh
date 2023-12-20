#!/bin/sh
agent_script=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

$agent_script \
    --footsies-self-play-snapshot-freq 1000 \
    --footsies-self-play-max-snapshots 100 \
    --footsies-self-play-mix-bot 50 \
    --footsies-self-play-port 11001 \
    "$@"