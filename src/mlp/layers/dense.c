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
{
    Dense *dense = self->impl;
    size_t output = dense->W->shape[0];
    size_t input = dense->W->shape[1];

    for (size_t i = 0; i < output; i++)
    {
        float sum = 0;

        for (size_t j = 0; j < input; j++)
        {
            size_t W_idx[2] = {i, j};
            size_t X_idx[2] = {j, 0};

            sum += tensor_get(dense->W, W_idx) * tensor_get(self->X, X_idx);
        }
        
        size_t Z_idx[2] = {i, 0};
        sum += tensor_get(dense->b, Z_idx);
        tensor_set(dense->Z, Z_idx, sum);
    }

    self->A = dense->Z;
}

static void layer_dense_backward(Layer *self)
static void layer_dense_free(Layer *self)
Layer *layer_dense_new(size_t input, size_t output)
