#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to --misc.no-load --agent.name curriculum_opp_dense_reward --agent.kwargs use_opponent_model True --env.kwargs dense_reward True "$@"