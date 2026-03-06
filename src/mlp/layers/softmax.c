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
 *  dZ = y_prediction - y_label
 *
 *  Note: Implementation for cross_entropy.c
 *
 * Shapes:
 *  Z: (n, batch)
 *  A: (n, batch)
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
    size_t n = Z->shape[0];
    size_t batch_size = Z->shape[1];

    Tensor *A = tensor_clone(Z);
    if (!A)
    {
        return;
    }

    // A_i = exp(Z_i - max(Z)) / (sum_j exp(Z_j - max(Z)))
    for (size_t i = 0; i < batch_size; i++)
    {
        // max(Z) for every batch
        float max = Z->data[i];

        for (size_t j = 0; j < n; j++)
        {
            size_t idx = j * batch_size + i;
            float curr = Z->data[idx];
            if (curr > max)
            {
                max = curr;
            }
        }

        // exponents
        float sum = 0;
        for (size_t j = 0; j < n; j++)
        {
            size_t idx = j * batch_size + i;
            A->data[idx] = expf(Z->data[idx] - max);
            sum += A->data[idx];
        }

        // normalization
        for (size_t j = 0; j < n; j++)
        {
            size_t idx = j * batch_size + i;
            A->data[idx] /= sum;
        }
    }

    if (self->output)
    {
        tensor_free(self->output);
    }

    self->output = A;
}

static void layer_softmax_backward(Layer *self)
{
    if (!self || !self->gradient_output)
    {
        return;
    }

    if (self->gradient_input)
    {
        tensor_free(self->gradient_input);
    }

    self->gradient_input = tensor_clone(self->gradient_output);
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