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