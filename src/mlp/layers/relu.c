#include "layer.h"

/**
 * @brief ReLU activation layer
 *
 * Forward input:
 *  X
 *
 * Forward computes:
 *  A = max(0, X)
 *
 * Backward input:
 *  dA = gradient_output
 *
 * Backward computes:
 *  dX = dA if X > 0
 *  dX = 0  if 0
 *
 * Shapes:
 *  X: (n, 1)
 *  A: (n, 1)
 */

typedef struct
{
    char _; // unused
} ReLU;

static void layer_relu_forward(Layer *self)
{
    if (!self || !self->input)
    {
        return;
    }

    Tensor *Z = self->input;

    // A = max(0, X)
    Tensor *A = tensor_clone(Z);
    if (!A)
    {
        return;
    }

    for (size_t i = 0; i < A->size; i++)
    {
        if (Z->data[i] < 0)
        {
            A->data[i] = 0;
        }
    }

    if (self->output)
    {
        tensor_free(self->output);
    }

    self->output = A;
}

static void layer_relu_backward(Layer *self)
{
    if (!self || !self->input || !self->gradient_output)
    {
        return;
    }

    Tensor *X = self->input;
    Tensor *dA = self->gradient_output;

    // dX = dA if X > 0
    // dX = 0  if 0
    Tensor *dX = tensor_clone(dA);
    if (!dX)
    {
        return;
    }

    for (size_t i = 0; i < dX->size; i++)
    {
        if (X->data[i] < 0)
        {
            dX->data[i] = 0;
        }
    }

    if (self->gradient_input)
    {
        tensor_free(self->gradient_input);
    }

    self->gradient_input = dX;
}

static void layer_relu_free(Layer *self)
{
    // holder, just for structure

    return;
}

Layer *layer_relu_new(void)
{
    ReLU *relu = calloc(1, sizeof(ReLU));
    if (!relu)
    {
        return NULL;
    }

    static const LayerOps ops = {
        .forward = layer_relu_forward,
        .backward = layer_relu_backward,
        .free = layer_relu_free};

    Layer *layer = layer_new(relu, &ops);
    if (!layer)
    {
        free(relu);

        return NULL;
    }

    return layer;
}