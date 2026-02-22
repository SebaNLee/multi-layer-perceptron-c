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
{
    Dense *dense = self->impl;

    tensor_free(dense->W);
    tensor_free(dense->b);
    tensor_free(dense->Z);
    tensor_free(dense->dW);
    tensor_free(dense->db);

    free(dense);
}

Layer *layer_dense_new(size_t input, size_t output)
{
    size_t W_shape[2] = {output, input};
    size_t b_shape[2] = {output, 1};

    Dense *dense = malloc(sizeof(Dense));
    if (!dense)
    {
        return NULL;
    }

    dense->W = tensor_new(2, W_shape);
    dense->b = tensor_new(2, b_shape);
    dense->Z = tensor_new(2, b_shape);
    dense->dW = tensor_new(2, W_shape);
    dense->db = tensor_new(2, b_shape);

    static const LayerOps ops = {
        .forward = layer_dense_forward,
        .backward = layer_dense_backward,
        .free = layer_dense_free
    };

    return layer_new(dense, &ops);
}