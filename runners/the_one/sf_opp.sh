#!/bin/sh
./runners/self-play.sh ./runners/footsies.sh to --agent.name sf_opp --agent.kwargs use_opponent_model True --agent.kwargs actor_entropy_coef 0.1 "$@"