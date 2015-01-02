#!/bin/bash
# (example)
# create a neural network with supplied weights and feed-forward some data.
# e.g.,
# ./run.sh /tmp/weights.bin test-data/iris.csv 

java -cp target/nn-light.jar net.parasec.nn.network.Runner $@ 

