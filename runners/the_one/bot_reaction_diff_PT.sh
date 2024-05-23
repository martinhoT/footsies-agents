#!/bin/sh
./runners/footsies.sh to --agent.name bot_reaction_diff_PT --misc.save --misc.no-load --misc.log-base-folder runs_ex --env.footsies-wrapper-simple.no-allow-agent-special-moves --agent.kwargs use_reaction_time True --agent.kwargs game_model_method differences --agent.kwargs one_decision_at_hitstop False --agent.kwargs learn all --seed 0 "$@"
