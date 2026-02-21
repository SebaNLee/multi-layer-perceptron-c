#include <stdlib.h>
#include "layer.h"

Layer *layer_create(void *impl, LayerOps *ops)
{
    if (!impl || !ops)
    {
        return NULL;
    }

    Layer *layer = malloc(sizeof(Layer));
    if (!layer)
    {
        return NULL;
    }

    layer->X = NULL;
    layer->A = NULL;
    layer->dA = NULL;
    layer->dX = NULL;

    layer->ops = ops;
    layer->impl = impl;

    return layer;
}

void layer_forward(Layer *layer, Tensor *input)
{
    if (!layer || !layer->ops || !layer->ops->forward)
    {
        return;
    }

    layer->X = input;
    layer->ops->forward(layer);
}

void layer_backward(Layer *layer, Tensor *grad_output)
{
    if (!layer || !layer->ops || !layer->ops->backward)
    {
        return;
    }

    layer->dA = grad_output;
    layer->ops->backward(layer);
}

void layer_free(Layer *layer)
{
    if (!layer)
    {
        return;
    }

    if (layer->ops && layer->ops->free)
    {
        layer->ops->free(layer);
    }

    layer->ops->free(layer);
    free(layer);
}
