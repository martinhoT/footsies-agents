#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to --misc.no-load --agent.name curriculum_opp_reaction_time --agent.kwargs use_opponent_model True --agent.kwargs use_reaction_time True "$@"