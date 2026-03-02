#include "mlp/optimizer/stochastic_gradient_descent.h"
#include "mlp/layers/dense.h"

/**
 * @brief Stochastic gradient descent optimizer
 *
 * W = W - learning_rate * dW
 * b = b - learning_rate * db
 *
 */
typedef struct
{
    float learning_rate;
} StochasticGradientDescent;

static void optimizer_stochastic_gradient_descent_step(Optimizer *self, MLP *mlp)
{
    if (!self || !mlp || !self->impl)
    {
        return;
    }

    StochasticGradientDescent *sgd = self->impl;

    for (size_t i = 0; i < mlp->layers_count; i++)
    {
        Layer *layer = mlp->layers[i];

        if (layer && layer->ops && layer->ops->apply_gradients)
        {
            layer->ops->apply_gradients(layer, sgd->learning_rate);
        }
    }
}

static void optimizer_stochastic_gradient_descent_free(Optimizer *self)
{
    if (!self)
    {
        return;
    }

    if (self->impl)
    {
        free(self->impl);
    }
}

Optimizer *optimizer_stochastic_gradient_descent_new(float learning_rate)
{
    StochasticGradientDescent *sgd = malloc(sizeof(StochasticGradientDescent));
    if (!sgd)
    {
        return NULL;
    }

    sgd->learning_rate = learning_rate;

    static const OptimizerOps ops = {
        .step = optimizer_stochastic_gradient_descent_step,
        .free = optimizer_stochastic_gradient_descent_free};

    Optimizer *optimizer = optimizer_new(sgd, &ops);
    if (!optimizer)
    {
        free(sgd);

        return NULL;
    }

    return optimizer;
}
