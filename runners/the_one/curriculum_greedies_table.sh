#!/bin/sh
./runners/curriculum.sh ./runners/footsies.sh to_no_specials --no-load -mB consider_explicit_opponent_policy true --name curriculum_greedies_table -mS critic_agent_update q_learning critic_opponent_update q_learning -mN critic_lr 0.1 -mB critic_table true