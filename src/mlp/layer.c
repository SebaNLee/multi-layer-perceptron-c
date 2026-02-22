#include <stdlib.h>
#include "layer.h"

Layer *layer_new(void *impl, LayerOps *ops)
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

    layer->input = NULL;
    layer->output = NULL;
    layer->gradient_output = NULL;
    layer->gradient_input = NULL;

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

    layer->input = input;
    layer->ops->forward(layer);
}

void layer_backward(Layer *layer, Tensor *grad_output)
{
    if (!layer || !layer->ops || !layer->ops->backward)
    {
        return;
    }

    layer->gradient_output = grad_output;
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

    free(layer);
}
