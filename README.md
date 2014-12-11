lightweight neural network
--------------------------

this is a work in progress.

features
--------
supervised learning with:
- stochastic backpropagation
- cross validation
- k-fold cross-validation

example
-------

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

