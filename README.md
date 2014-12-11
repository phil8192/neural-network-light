lightweight neural network
--------------------------

this is a work in progress.

features
--------
supervised learning with:
- stochastic backpropagation
- cross validation
- k-fold cross-validation

iris classification example
---------------------------
see cross-validation.sh

```bash
phil@Eris:~/neural-network-light$ ./cross-validation.sh
[1418301930132 DataLoader main] loaded 150 training instances in 6ms.
[1418301930133 Data main] splitting data into training/testing subsets
[1418301930134 ANN main] initialising network with structure: [4, 10, 3]
[1418301930241 Train main] training complete. best epoch = 1000 training mse = 0.17841 testing mse = 0.35683 testing ae = 0.02 min = 0.00 max = 0.09
[1418301930241 Train main] training took 107ms.
[1418301930242 Report main] dumping errors to: reporter/model/errors.csv
[1418301930377 Report main] wrote data in 135ms
graph in reporter/model/error.png
```

![alt tag](https://raw.githubusercontent.com/phil8192/neural-network-light/master/reporter/model/errors.png)

training options
----------------
```
--file=<dataset.csv>        location of dataset
--output_nodes=<#>          number of network output nodes
--holdback=<0:1>            ratio of data to holdback for test set
--k=<0:n>                   number of k-folds k<=1 disable. max=dataset_len
--min_weight=<r>            min weight for random weight initialisation
--max_weight=<r>            max weight for random weight initialisation
--learning_rate=<r>         learning rate
--momentum=<r>              momentum term
--epochs=<n>                number of training epochs
--model_output=<dir>        save learned weights + reports to dir 
--hidden_nodes=<n1,n2..,N>  number of _hidden_ nodes in each layer n
```

general example
---------------

```bash
./train.sh --file=some-data.csv --output_nodes=3 --holdback=0.2 --k=0 \
            --min_weight=-0.5 --max_weight=0.5 --learning_rate=0.1 \ 
            --momentum=0.25 --epochs=1000 --model_output=/tmp \
            --hidden_nodes=3,2
```

Will train a neural network on some data. 20% of the data will be
reserved for a test validation subset; no k-folding will be performed.
the network structure will be:

```
(i1)
    \    /> (l1,1)\             /> (ol,1)
(i2)\\ //         \\ /> (l2,1) / 
      X---> (l1,2)--X         X--> (ol,2)
(i3)// \\         // \> (l2,2) \
    /    \> (l1,3)/             \> (ol,3)
(i4)                              
                                    ^--- output layer: 3 neurons             
                         ^-------------- layer 2: 2 neurons 
             ^-------------------------- layer 2: 3 neurons
 ^-------------------------------------- input layer: 4 nodes
```

Where each layer is fully interconnected. Note that the number of nodes in 
the input layer, and neurons in the output layer are automatically determined 
by the number of columns in the training data: in this case, there are 4 input 
features (first 4 columns) and 3 output classes (last 3 columns). The number of 
output neurons is determined by the output_nodes flag.

