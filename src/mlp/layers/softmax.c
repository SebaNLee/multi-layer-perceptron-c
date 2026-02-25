#include "layer.h"
#include <math.h>

/**
 * @brief Softmax activation layer
 *
 * Forward input:
 *  X
 *
 * Forward computes:
 *  A_i = exp(X_i - max(X)) / (sum_j exp(X_j - max(X)))
 *
 *  Note: Substracting max() to prevent overflows
 *
 * Backward:
 *  Not implemented
 *  When combined with cross entropy loss, the gradient should be: dZ = prediction - value
 *
 * Shapes:
 *  X: (n, 1)
 *  A: (n, 1)
 */
typedef struct
{
    char _; // unused
} Softmax;

static void layer_softmax_forward(Layer *self)
{
}

static void layer_softmax_backward(Layer *self)
{
}

static void layer_softmax_free(Layer *self)
{
    // holder, just for structure

    return;
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
        .free = layer_softmax_free};

    Layer *layer = layer_new(softmax, &ops);
    if (!layer)
    {
        free(softmax);

        return NULL;
    }

    return layer;
}