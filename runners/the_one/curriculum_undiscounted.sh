#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to --misc.no-load --agent.kwargs critic_opponent_update q_learning --agent.name curriculum_undiscounted --agent.kwargs critic_discount 1.0