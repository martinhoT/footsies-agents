#!/bin/sh
python3 main.py brisket --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 --footsies-wrapper-norm --footsies-wrapper-acd -mN epsilon 0 min_epsilon 0 --episodes 100 --no-save --no-log "$@"