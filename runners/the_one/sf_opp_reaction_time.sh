#!/bin/sh
./runners/self-play.sh ./runners/footsies.sh to --agent.name sf_opp_reaction_time --agent.kwargs use_opponent_model True --agent.kwargs use_reaction_time True "$@"