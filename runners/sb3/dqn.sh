#!/bin/sh
./runners/footsies.sh sb3.dqn \
    --time-steps 10000000 \
    --agent.name sb3_dqn \
    --agent.kwargs learning_rate 0.0001 \
    --agent.kwargs gamma 0.9 \
    --agent.kwargs batch_size 64 \
    "$@"