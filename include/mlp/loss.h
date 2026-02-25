#include "tensor.h"

typedef struct
{
    float (*forward)(Loss *self, const Tensor *y_prediction, const Tensor *y_label);
    Tensor *(*backward)(Loss *self, const Tensor *y_prediction, const Tensor *y_label);
    void (*free)(Loss *self);
} LossOps;

typedef struct
{
    LossOps *ops;
    void *impl;
} Loss;
