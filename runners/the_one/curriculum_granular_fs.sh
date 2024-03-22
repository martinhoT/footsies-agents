#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to_no_specials --misc.no-load --agent.kwargs critic_opponent_update q_learning --agent.name curriculum_granular_fs --agent.kwargs accumulate_at_frameskip False
