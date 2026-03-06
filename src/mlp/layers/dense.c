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
 *  db = sum (dZ, axis=batch_size)
 *  dX = W^T dZ
 *
 * Applies gradients:
 *  W = W - learning_rate * dW
 *  b = b - learning_rate * db
 *
 * Shapes:
 *  W: (output, input)
 *  b: (output, 1)
 *  X: (input, batch)
 *  Z: (output, batch)
 *  dW: (output, input)
 *  db: (output, 1)
 *  dX: (input, batch)
 *  dZ: (output, batch)
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
    size_t output_size = W->shape[0];
    size_t batch_size = X->shape[1];

    // Z = W X + b
    Tensor *WX = tensor_matrix_multiplication(W, X);
    if (!WX)
    {
        return;
    }

    Tensor *Z = tensor_new(2, WX->shape);
    if (!Z)
    {
        tensor_free(WX);
        return;
    }

    for (size_t i = 0; i < output_size; i++)
    {
        for (size_t j = 0; j < batch_size; j++)
        {
            size_t idx = i * batch_size + j;
            Z->data[idx] = WX->data[idx] + b->data[i];
        }
    }

    tensor_free(WX);

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
    size_t output_size = dZ->shape[0];
    size_t batch_size = dZ->shape[1];

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

    // db = sum (dZ, axis=batch_size)
    size_t db_shape[2] = {output_size, 1};
    Tensor *db = tensor_new(2, db_shape);
    if (!db)
    {
        tensor_free(dX);
        tensor_free(dW);
        return;
    }

    for (size_t i = 0; i < output_size; i++)
    {
        float sum = 0;

        for (size_t j = 0; j < batch_size; j++)
        {
            sum += dZ->data[i * batch_size + j]; // TODO use tensor api (less efficiency)
        }

        db->data[i] = sum;
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

static void layer_dense_apply_gradients(Layer *self, float learning_rate)
{
    if (!self || !self->impl)
    {
        return;
    }

    Dense *dense = self->impl;

    if (!dense->W || !dense->dW || !dense->b || !dense->db)
    {
        return;
    }

    for (size_t i = 0; i < dense->W->size; i++)
    {
        dense->W->data[i] -= learning_rate * dense->dW->data[i];
    }

    for (size_t i = 0; i < dense->b->size; i++)
    {
        dense->b->data[i] -= learning_rate * dense->db->data[i];
    }
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
        .apply_gradients = layer_dense_apply_gradients,
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