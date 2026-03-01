#ifndef MLP_OPTIMIZER_H
#define MLP_OPTIMIZER_H

#include "mlp/mlp.h"

typedef struct Optimizer Optimizer;

typedef struct
{
    void (*step)(Optimizer *self, MLP *mlp);
    void (*free)(Optimizer *self);
} OptimizerOps;

struct Optimizer
{
    const OptimizerOps *ops;
    void *impl;
};

// generic layer API, calls corresponding OptimizerOps *
Optimizer *optimizer_new(void *impl, const OptimizerOps *ops);
void optimizer_step(Optimizer *optimizer, MLP *mlp);
void optimizer_free(Optimizer *optimizer);

#endif