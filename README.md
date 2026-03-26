TODO

# Multi Layer Perceptron C

This repository implements a Multilayer Perceptron (MLP) library from scratch in C, without the use of external libraries. The implementation covers core concepts in neural networks, such as forward propagation, backpropagation, loss calculation, and parameter optimization. The library is designed in a modular way, allowing flexible declaration of layers, activation functions, loss functions and other configurations.

The design and general structure are primarily based on *Learning Representations by Back-Propagating Errors* (1986) by David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams [[1]] [[2]].

[1]: https://www.nature.com/articles/323533a0
[2]: https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf


## Lore



(lore)

## Scripts Usage

Detailed below are implemented examples that use the library with the specified datasets and the results obtained. The ```run.sh``` shell script automates their execution, and ```dataset.sh``` handles the download of the datasets.

You can use the shell scrips like this:

```
$ ./run.sh [release|debug] <example>
```

```
$ ./datasets.sh [all|emnist|mushroom|meteorite|engine]
```

<!-- TODO compile only library binary

```
$ ./datasets.sh library
``` -->

## Results

#### EMNIST Digits

```
MLP:  784 -> 128(Sigmoid) -> 64(Sigmoid) -> 32(ReLU) -> 10(Softmax)
Loss: Cross Entropy
SGD:  0.1

Results:
Model Accuracy                :  98.64500% (39458/40000)
Average True Class Confidence :  97.99243%
Mean Test Loss                :   0.04546
Test Samples                  : 40000/40000

Runtime: 117.204315s
```

#### EMNIST Letters

```
MLP:  784 -> 256(Sigmoid) -> 128(ReLU) ->26(Softmax)
Loss: Cross Entropy
SGD:  0.1

Results:
Model Accuracy                :  89.94231% (18708/20800)
Average True Class Confidence :  85.67694%
Mean Test Loss                :   0.31822
Test Samples                  : 20800/20800

Runtime: 134.920381s
```

#### NASA Meteorite Landings

```
MLP:  4 -> 256(ReLU) -> 256(Sigmoid) -> 256(Sigmoid) -> 2(Softmax)
Loss: Cross Entropy
SGD:  0.1

Results:
Model Accuracy                :  99.75550% (408/409)
Average True Class Confidence :  98.77888%
Mean Test Loss                :   0.01334
Test Samples                  : 409/2042

Runtime: 4.003218s
```

#### NASA CMAPSS Jet Engine Simulated Data

```
MLP:  24 -> 256(ReLU) -> 128(ReLU) -> 1(Sigmoid)
Loss: Mean Squared Error
SGD:  0.1

Results:
Model Accuracy                :  60.77053% (2508/4127) (with |error| <= 10%)
Average True Class Confidence :  67.14149%
Mean Test Loss                :   0.01065
Test Samples                  : 4127/20631

Runtime: 7.771604s
```

#### UCI Mushroom

```
MLP:  127 -> 128(ReLU) -> 64(ReLU) -> 2(Softmax)
Loss: Cross Entropy
SGD:  0.05

Results:
Model Accuracy                : 100.00000% (1625/1625)
Average True Class Confidence :  99.97532%
Mean Test Loss                :   0.00025
Test Samples                  : 1625/8124

Runtime: 1.233496s
```

#### XOR

```
MLP:  2 -> 8(ReLU) -> 2(Softmax)
Loss: Cross Entropy
SGD:  0.1

Results:
Model Accuracy                : 100.00000% (4/4)
Average True Class Confidence :  99.82397%
Mean Test Loss                :   0.00176
Test Samples                  : 4/4

Runtime: 0.011155s
```

## Devs

To standardize code format, there is a pre-commit hook that uses Clang when committing. It can be installed with: 

```
pip install pre-commit
pre-commit install
```

For manual compilation and execution of the generated Makefiles and example binaries, run:

```
build$ cmake ..
build$ make
$ ./build/examples/<binary>
```

## Library Usage

(contract)

## Reference Materials

#### Main references

Learning Representations by Back-Propagating Errors (1986) by David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams [[1](https://www.nature.com/articles/323533a0)] [[2](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)]

Multilayer Perceptron, Explained: A Visual Guide with Mini 2D Dataset [[Medium](https://medium.com/data-science/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c)]

#### More references

Neural Network and Deep Learning, Richard Nielsen [[1](http://neuralnetworksanddeeplearning.com/index.html)]

Building a Multi-Layer Perceptron from Scratch with NumPy [[Medium](https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d)]

Xavier and He Normal (He-et-al) Initialization [[Medium](https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)]

Neural networks [[YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)]

Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) [[YouTube](https://www.youtube.com/watch?v=w8yWXqWQYmU)]

Neural-Network-from-scratch-without-TensorFlow-PyTorch [[GitHub](https://github.com/OmBaval/Neural-Network-from-scratch-without-TensorFlow-PyTorch)]

Deep Learning With PyTorch — Tensor Basics: Stride, Offset, Contiguous Tensors [[Medium](https://medium.com/swlh/deep-learning-with-pytorch-tensor-basics-part-1-stride-offset-contiguous-tensors-5d87476b7d9f)]

#### Datasets

EMNIST [[NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)]

NASA Meteorite Landings [[NASA Open Data Portal](https://data.nasa.gov/dataset/meteorite-landings)]

NASA CMAPSS Jet Engine Simulated Data [[NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)]

UCI Mushroom [[UCI](https://archive.ics.uci.edu/dataset/73/mushroom)]
