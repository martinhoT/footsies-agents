#!/bin/sh
python3 main.py autoencoder \
    --footsies-path ../Footsies-Gym/Build/FOOTSIES.x86_64 \
    --footsies-wrapper-norm \
    --footsies-wrapper-acd \
    -eB by_example true \
    -mN \
        encoded_dim 36 \
        encoder_hidden_layers 1 \
        encoder_hidden_layer_size 32 \
        decoder_hidden_layers 1 \
        decoder_hidden_layer_size 32 \
    -mB \
        normalized false \
        include_sequentiality_loss false \
    "$@"