#include "tensor.h"

typedef struct
{
    // TODO
    void (*forward)(Layer * self);
    void (*backward)(Layer * self);
    void (*free)(Layer * self);
} LayerOps;

typedef struct
{
    // Z = W X + b
    // A = phi(Z)

    Tensor *X; // input
    Tensor *A; // output

    Tensor *dA; // TODO
    Tensor *dX; // TODO

    LayerOps * ops;
    void * impl;
} Layer;

// generic layer API, calls corresponding LayerOps *
Layer *layer_create(void *impl, LayerOps *ops);
void layer_forward(Layer *layer, Tensor *input);
void layer_backward(Layer *layer, Tensor *grad_output);
void layer_free(Layer *layer);