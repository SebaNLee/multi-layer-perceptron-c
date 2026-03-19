TODO

## Multi Layer Perceptron C

<!-- This repo implements a Multilayer Perceptron library made from scratch in C. Should work in any machine as it is native C without usage of external libraries. -->

## Lore



(lore)

## Scripts Usage

Detailed below are implemented examples that uses the library with the specified datasets and the results obtained. The ```run.sh``` shell scripts automates the execution of these and ```dataset.sh``` handles the download of the datasets.

You can use the shell scrips like this:

```
$ ./run.sh [release|debug] <example>
```

```
$ ./datasets.sh [all|emnist|mushroom|meteorite|engine]
```

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

## Library Usage

(contract)


## For devs

pip install pre-commit
pre-commit install
(automatically applies clang formatting and adds it to the staging area)

CMake:
    cd build
    cmake ..
    make
    <execute binary path>

## Reference Materials

