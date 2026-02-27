#include "mlp/tensor.h"

typedef struct Layer Layer;

typedef struct
{
    void (*forward)(Layer *self);
    void (*backward)(Layer *self);
    void (*free)(Layer *self);
} LayerOps;

struct Layer
{
    Tensor *input;
    Tensor *output;

    Tensor *gradient_input;
    Tensor *gradient_output;

    LayerOps *ops;
    void *impl;
};

// generic layer API, calls corresponding LayerOps *
Layer *layer_new(void *impl, LayerOps *ops);
void layer_forward(Layer *layer, Tensor *input);
void layer_backward(Layer *layer, Tensor *grad_output);
void layer_free(Layer *layer);