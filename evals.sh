#!/bin/zsh
python3 -m scripts.evaluation.all --processes 12 | tee evaluation_all`date +%d-%m-%Y_%H-%M-%S`.log
