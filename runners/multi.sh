#!/bin/sh
agent_script=$1
shift 1 # Remove the first argument, which is the path to the wrapped script, so that the rest of the arguments are passed to it

$agent_script \
    "$@" \
    --misc.hogwild \
    --misc.hogwild-cpus 12 \
    --misc.hogwild-n-workers 6
