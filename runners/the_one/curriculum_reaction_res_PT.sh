#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to --agent.name curriculum_reaction_res_PT --misc.save --misc.no-load --env.footsies-wrapper-simple.no-allow-agent-special-moves --agent.kwargs use_reaction_time True --agent.kwargs game_model_method residual --agent.kwargs one_decision_at_hitstop False --seed 0 "$@"
