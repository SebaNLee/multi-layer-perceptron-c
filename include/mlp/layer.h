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

    Tensor *dA;
    Tensor *dX;

    LayerOps * ops;
    void * impl;
} Layer;