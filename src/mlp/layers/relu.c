#include "layer.h"

/**
 * @brief ReLU activation layer
 *
 * Forward input:
 *  Z
 *
 * Forward computes:
 *  A_i = max(0, Z_i)
 *
 * Backward input:
 *  dA = gradient_output
 *
 * Backward computes:
 *  dZ_i = dA_i if Z_i > 0
 *  dZ_i = 0    otherwise
 *
 * Shapes:
 *  Z: (n, 1)
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

    // A_i = max(0, Z_i)
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

    Tensor *Z = self->input;
    Tensor *dA = self->gradient_output;

    // dZ_i = dA_i if Z_i > 0
    // dZ_i = 0    otherwise
    Tensor *dX = tensor_clone(dA);
    if (!dX)
    {
        return;
    }

    for (size_t i = 0; i < dX->size; i++)
    {
        if (Z->data[i] < 0)
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