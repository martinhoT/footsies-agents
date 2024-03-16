#!/bin/sh
agent_script=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

$agent_script \
    --footsies-curriculum \
    --wrapper-time-limit 600 \
    --footsies-wrapper-adv \
    -eS \
        sync_mode synced_non_blocking \
    "$@"