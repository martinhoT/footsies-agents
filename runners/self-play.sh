#!/bin/sh
agent_script=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

$agent_script \
    "$@" \
    --self-play.enabled \
    --self-play.max-opponents 10 \
    --self-play.snapshot-interval 2000 \
    --self-play.switch-interval 500 \
    --self-play.mix-bot 1 \
    --env.wrapper-time-limit 1000 \
    --env.kwargs \
        sync_mode synced_non_blocking
