#!/bin/bash
# get Fisher's famous iris dataset from:
# https://archive.ics.uci.edu/ml/datasets/Iris
#
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. class: 
# -- Iris Setosa 
# -- Iris Versicolour 
# -- Iris Virginica
#
# the class is converted to 0.9, 0.1, 0.1 for setosa, 0.1, 0.9, 0.1 for
# versicolour and 0.1, 0.1, 0.9 for virginica; so that the neural net will have
# 3 outputs, each of which corresponds to the class. 0.9 is used instead of 1.0
# and 0.1 instead of 0.0 because a sigmoid can never reach 0 or 1.
#
# use: ./iris.sh >iris.csv
curl -s https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data \
  |sed 's/Iris-setosa/0.9,0.1,0.1/; s/Iris-versicolor/0.1,0.9,0.1/; s/Iris-virginica/0.1,0.1,0.9/'

