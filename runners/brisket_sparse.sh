#!/bin/sh
python3 main.py brisket --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 --footsies-wrapper-norm --footsies-wrapper-acd --model-name brisket_sparse -eB dense_reward False "$@"