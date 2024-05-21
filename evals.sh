#!/bin/zsh
python3 -m scripts.evaluation.all --seeds 6 --processes 12 --reverse | tee evaluation_all`date +%d-%m-%Y_%H-%M-%S`.log

