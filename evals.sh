#!/bin/zsh
python3 -m scripts.evaluation.all --seeds 1 --processes 4 | tee evaluation_all`date +%d-%m-%Y_%H-%M-%S`.log

