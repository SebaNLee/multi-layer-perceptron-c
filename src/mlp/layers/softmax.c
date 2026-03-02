#include "mlp/layers/softmax.h"
#include <math.h>

/**
 * @brief Softmax activation layer
 *
 * Forward input:
 *  Z
 *
 * Forward computes:
 *  A_i = exp(Z_i - max(Z)) / (sum_j exp(Z_j - max(Z)))
 *
 *  Note: Substracting max() to prevent overflows
 *
 * Backward:
 *  Not implemented
 *  When combined with cross entropy loss, the gradient should be: dZ = prediction - value
 *
 * Shapes:
 *  Z: (n, 1)
 *  A: (n, 1)
 */
typedef struct
{
    char _; // unused
} Softmax;

static void layer_softmax_forward(Layer *self)
{
    if (!self || !self->input)
    {
        return;
    }

    Tensor *Z = self->input;
    Tensor *A = tensor_clone(Z);
    if (!A)
    {
        return;
    }

    // max(Z)
    float max = Z->data[0];
    for (size_t i = 0; i < Z->size; i++)
    {
        if (Z->data[i] > max)
        {
            max = Z->data[i];
        }
    }

    // A_i = exp(Z_i - max(Z)) / (sum_j exp(Z_j - max(Z)))
    float sum = 0;
    for (size_t i = 0; i < A->size; i++)
    {
        A->data[i] = expf(Z->data[i] - max);
        sum += A->data[i];
    }

    for (size_t i = 0; i < A->size; i++)
    {
        A->data[i] /= sum;
    }

    if (self->output)
    {
        tensor_free(self->output);
    }

    self->output = A;
}

static void layer_softmax_backward(Layer *self)
{
}

static void layer_softmax_free(Layer *self)
{
    if (!self)
    {
        return;
    }

    if (self->impl)
    {
        free(self->impl);
    }
}

Layer *layer_softmax_new(void)
{
    Softmax *softmax = calloc(1, sizeof(Softmax));
    if (!softmax)
    {
        return NULL;
    }

    static const LayerOps ops = {
        .forward = layer_softmax_forward,
        .backward = layer_softmax_backward,
        .apply_gradients = NULL,
        .free = layer_softmax_free};

    Layer *layer = layer_new(softmax, &ops);
    if (!layer)
    {
        free(softmax);

        return NULL;
    }

    return layer;
}