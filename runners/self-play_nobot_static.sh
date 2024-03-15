#!/bin/sh
agent_script=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

$agent_script \
    --footsies-self-play \
    --footsies-self-play-max-snapshots 1 \
    --footsies-self-play-snapshot-interval 1000000 \
    --footsies-self-play-switch-interval 1000000 \
    --footsies-self-play-mix-bot 0 \
    --wrapper-time-limit 1000 \
    --penalize-truncation -1 \
    -eS \
        sync_mode synced_non_blocking \
    "$@"