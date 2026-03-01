#include "mlp/layers/dense.h"

/**
 * @brief Fully connected linear layer
 *
 * Foward input:
 *  X
 *
 * Forward computes:
 *  Z = W X + b
 *
 * Backward input:
 *  dZ = gradient_output
 *
 * Backward computes:
 *  dW = dZ X^T
 *  db = dZ (batch size = 1) // TODO
 *  dX = W^T dZ
 *
 * Shapes:
 *  W: (output, input)
 *  b: (output, 1)
 *  X: (input, 1)
 *  Z: (output, 1)
 */
typedef struct
{
    Tensor *W; // weights
    Tensor *b; // biases

    Tensor *Z; // output (before applying activation function)

    Tensor *dW; // weights gradient
    Tensor *db; // biases gradient
} Dense;

static void layer_dense_forward(Layer *self)
{
    if (!self || !self->impl)
    {
        return;
    }

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
    if (!self || !self->impl)
    {
        return;
    }

    Dense *dense = self->impl;
    Tensor *X = self->input;
    Tensor *dZ = self->gradient_output;
    Tensor *W = dense->W;

    // dW = dZ X^T
    Tensor *X_transposed = tensor_transpose_2d(X);
    if (!X_transposed)
    {
        return;
    }

    Tensor *dW = tensor_matrix_multiplication(dZ, X_transposed);
    tensor_free(X_transposed);
    if (!dW)
    {
        return;
    }

    // dX = W^T dZ
    Tensor *W_transposed = tensor_transpose_2d(W);
    if (!W_transposed)
    {
        tensor_free(dW);
        return;
    }

    Tensor *dX = tensor_matrix_multiplication(W_transposed, dZ);
    tensor_free(W_transposed);
    if (!dX)
    {
        tensor_free(dW);
        return;
    }

    // !!
    // !! TODO batch management
    // !!
    // db = dZ (batch size = 1)
    Tensor *db = tensor_clone(dZ);
    if (!db)
    {
        tensor_free(dX);
        tensor_free(dW);
        return;
    }

    // assign results
    if (dense->dW)
    {
        tensor_free(dense->dW);
    }

    if (self->gradient_input)
    {
        tensor_free(self->gradient_input);
    }

    if (dense->db)
    {
        tensor_free(dense->db);
    }

    dense->dW = dW;
    self->gradient_input = dX;
    dense->db = db;
}

static void layer_dense_free(Layer *self)
{
    Dense *dense = self->impl;

    tensor_free(dense->W);
    tensor_free(dense->b);
    tensor_free(dense->dW);
    tensor_free(dense->db);

    free(dense);
}

Layer *layer_dense_new(size_t input, size_t output, DenseInit init)
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

    switch (init)
    {
    case DENSE_INIT_HE:
        tensor_fill_he(dense->W);
        break;
    case DENSE_INIT_XAVIER:
        tensor_fill_xavier(dense->W);
        break;
    default:
        tensor_fill_xavier(dense->W); // default fallback
        break;
    }

    tensor_fill(dense->b, 0);

    static const LayerOps ops = {
        .forward = layer_dense_forward,
        .backward = layer_dense_backward,
        .free = layer_dense_free};

    Layer *layer = layer_new(dense, &ops);
    if (!layer)
    {
        tensor_free(dense->W);
        tensor_free(dense->b);
        free(dense);

        return NULL;
    }

    return layer;
}