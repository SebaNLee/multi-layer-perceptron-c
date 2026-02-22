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
static void layer_dense_forward(Layer *self)
static void layer_dense_backward(Layer *self)
static void layer_dense_free(Layer *self)
Layer *layer_dense_new(size_t input, size_t output)
