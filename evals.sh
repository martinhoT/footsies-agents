#!/bin/zsh
python3 -m scripts.evaluation.all --seeds 3 --processes 12 | tee evaluation_all`date +%d-%m-%Y_%H-%M-%S`.log

