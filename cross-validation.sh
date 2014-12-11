#!/bin/bash
# trainining with cross-validation example using iris dataset.

# train a 4-10-3 network (1 hidden layer with 10 neurons, 1 output layer with
# 3 neurons. 
# reserve 20% of the 45 instances for a test validation subset.
# train with 0.1 learning rate and 0.25 momentum.
# train for a maximum of 1000 epochs. 
./train.sh --file=test-data/iris.csv --output_nodes=3 --holdback=0.2 --k=0 \
           --min_weight=-1 --max_weight=1 --learning_rate=0.1 \
           --momentum=0.25 --epochs=1000 --model_output=reporter/model \
           --hidden_nodes=10

# plot an ascii graph of the training and testing mean squared error for each
# epoch. (1 epoch = backpropogation on all training examples).
R --vanilla < reporter/report.r >/dev/null
echo "graph in reporter/error.png"

