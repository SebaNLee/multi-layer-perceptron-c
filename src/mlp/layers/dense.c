#include "layer.h"

typedef struct
{
    // Z = W X + b
    // A = phi(Z)

    // dZ = dA
    // dW = dZ X^T
    // db = dZ
    // dX = W^T dZ

    Tensor *W; // weights
    Tensor *b; // biases

    Tensor *Z; // output (before applying activation function)

    Tensor *dW; // TODO
    Tensor *db; // TODO
} Dense;

static void layer_dense_forward(Layer *self)
{
    Dense *dense = self->impl;
    Tensor *X = self->input;
    Tensor *W = dense->W;
    Tensor *b = dense->b;

    // Z = W X + b
    Tensor *WX = tensor_matrix_multiplication(W, X);
    if (!WX)
    {
        return;
    }

    Tensor *Z = tensor_add(WX, b);
    tensor_free(WX);
    if (!Z)
    {
        return;
    }

    if (dense->Z)
    {
        tensor_free(dense->Z);
    }

    dense->Z = Z;
    self->output = Z;
}

static void layer_dense_backward(Layer *self)
{
}

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

    Dense *dense = calloc(1, sizeof(Dense));
    if (!dense)
    {
        return NULL;
    }

    dense->W = tensor_new(2, W_shape);
    dense->b = tensor_new(2, b_shape);
    if (!dense->W || !dense->b)
    {
        tensor_free(dense->W);
        tensor_free(dense->b);
        free(dense);

        return NULL;
    }

    static const LayerOps ops = {
        .forward = layer_dense_forward,
        .backward = layer_dense_backward,
        .free = layer_dense_free};

    return layer_new(dense, &ops);
}