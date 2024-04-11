#!/bin/sh
./runners/footsies_profiled.sh to --agent.name f_opp_reaction_time --agent.kwargs use_opponent_model True --agent.kwargs use_reaction_time True --agent.kwargs use_game_model True "$@"