#include "mlp/loss.h"

Loss *loss_new(void *impl, const LossOps *ops)
{
    if (!impl || !ops)
    {
        return NULL;
    }

    Loss *loss = malloc(sizeof(Loss));
    if (!loss)
    {
        return NULL;
    }

    loss->impl = impl;
    loss->ops = ops;

    return loss;
}

float loss_forward(Loss *loss, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!loss || !loss->ops || !loss->ops->forward)
    {
        return 0;
    }

    return loss->ops->forward(loss, y_prediction, y_label);
}

Tensor *loss_backward(Loss *loss, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!loss || !loss->ops || !loss->ops->backward)
    {
        return NULL;
    }

    return loss->ops->backward(loss, y_prediction, y_label);
}

void loss_free(Loss *loss)
{
    if (!loss)
    {
        return;
    }

    if (loss->ops && loss->ops->free)
    {
        loss->ops->free(loss);
    }

    free(loss);
}
