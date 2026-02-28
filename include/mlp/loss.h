#ifndef MLP_LOSS_H
#define MLP_LOSS_H

#include "mlp/tensor.h"

typedef struct Loss Loss;

typedef struct
{
    float (*forward)(Loss *self, const Tensor *y_prediction, const Tensor *y_label);
    Tensor *(*backward)(Loss *self, const Tensor *y_prediction, const Tensor *y_label);
    void (*free)(Loss *self);
} LossOps;

struct Loss
{
    LossOps *ops;
    void *impl;
};

#endif