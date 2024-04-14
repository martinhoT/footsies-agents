#!/bin/sh
./runners/footsies.sh sb3.a2c \
    --time-steps 10000000 \
    --agent.name sb3_a2c \
    --agent.kwargs learning_rate 0.0001 \
    --agent.kwargs gamma 0.9 \
    --agent.kwargs normalize_advantage False \
    --agent.kwargs ent_coef 0.04 \
    --agent.kwargs max_grad_norm 0.5 \
    --agent.kwargs use_sde False \
    --agent.kwargs gae_lambda 1.0 \
    "$@"