#include "layer.h"

typedef struct
{
    // Z = W X + b
    // A = phi(Z)

    // dZ = dA
    // dW = dZ dot X^T
    // db = dZ
    // dX = W^T dot dZ

    Tensor *W; // weights
    Tensor *b; // biases

    Tensor *Z; // output (before applying activation function)

    Tensor *dW; // TODO
    Tensor *db; // TODO
} Dense;
