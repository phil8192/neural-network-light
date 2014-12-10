#!/bin/bash
# test
# ~14.3s.
./train.sh --file=test-data/iris.csv --output_nodes=3 --holdback=0 --k=0 \
           --min_weight=-0.5 --max_weight=0.5 --learning_rate=0.1 \
           --momentum=0.25 --epochs=200000 --model_output=/tmp \
           --hidden_nodes=3,2,3

