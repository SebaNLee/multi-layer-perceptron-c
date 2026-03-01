#include "mlp/optimizer.h"
#include <stdlib.h>

Optimizer *optimizer_new(void *impl, const OptimizerOps *ops)
{
    if (!impl || !ops)
    {
        return NULL;
    }

    Optimizer *optimizer = malloc(sizeof(Optimizer));
    if (!optimizer)
    {
        return NULL;
    }

    optimizer->impl = impl;
    optimizer->ops = ops;

    return optimizer;
}

void optimizer_step(Optimizer *optimizer, MLP *mlp)
{
    if (!optimizer || !mlp)
    {
        return;
    }

    optimizer->ops->step(optimizer, mlp);
}

void optimizer_free(Optimizer *optimizer)
{
    if (!optimizer)
    {
        return;
    }

    if (optimizer->ops && optimizer->ops->free)
    {
        optimizer->ops->free(optimizer);
    }

    free(optimizer);
}
