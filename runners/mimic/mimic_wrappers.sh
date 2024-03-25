#!/bin/sh
./runners/footsies.sh mimic \
    --env.kwargs \
        by_example True \
    --misc.log-frequency 5000 \
    --misc.no-load \
    --env.no-footsies-wrapper-norm-guard \
    --env.footsies-wrapper-phasic \
    --env.footsies-wrapper-history \
        p1 True \
        p2 True \
        p1_n 5 \
        p2_n 5 \
        p1_distinct True \
        p2_distinct True \
    --agent.name mimic_wrappers
