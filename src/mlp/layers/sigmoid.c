#include "mlp/layers/sigmoid.h"
#include <math.h>

/**
 * @brief Sigmoid activation layer
 *
 * Forward input:
 *  Z
 *
 * Forward computes:
 *  A_i = 1 / (1 + e^(-Z_i))
 *
 * Backward input:
 *  dA = gradient_output
 *
 * Backward computes:
 *  dZ_i = dA_i * A_i * (1 - A_i)
 *
 * Shapes:
 *  Z: (n, 1)
 *  A: (n, 1)
 */
typedef struct
{
    char _; // unused
} Sigmoid;

static void layer_sigmoid_forward(Layer *self)
{
    if (!self || !self->input)
    {
        return;
    }

    Tensor *Z = self->input;

    // A_i = 1 / (1 + e^(-Z_i))
    Tensor *A = tensor_clone(Z);
    if (!A)
    {
        return;
    }

    for (size_t i = 0; i < A->size; i++)
    {

        A->data[i] = 1 / (1 + expf(-Z->data[i]));
    }

    if (self->output)
    {
        tensor_free(self->output);
    }

    self->output = A;
}

static void layer_sigmoid_backward(Layer *self)
{
    if (!self || !self->output || !self->gradient_output)
    {
        return;
    }

    Tensor *A = self->output;
    Tensor *dA = self->gradient_output;

    // dZ_i = dA_i * A_i * (1 - A_i)
    Tensor *dX = tensor_clone(dA);
    if (!dX)
    {
        return;
    }

    for (size_t i = 0; i < dX->size; i++)
    {
        dX->data[i] = dX->data[i] * A->data[i] * (1 - A->data[i]);
    }

    if (self->gradient_input)
    {
        tensor_free(self->gradient_input);
    }

    self->gradient_input = dX;
}

static void layer_sigmoid_free(Layer *self)
{
    // holder, just for structure

    return;
}

Layer *layer_sigmoid_new(void)
{
    Sigmoid *sigmoid = calloc(1, sizeof(Sigmoid));
    if (!sigmoid)
    {
        return NULL;
    }

    static const LayerOps ops = {
        .forward = layer_sigmoid_forward,
        .backward = layer_sigmoid_backward,
        .free = layer_sigmoid_free};

    Layer *layer = layer_new(sigmoid, &ops);
    if (!layer)
    {
        free(sigmoid);

        return NULL;
    }

    return layer;
}