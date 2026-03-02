#ifndef MLP_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_H
#define MLP_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_H

#include "mlp/optimizer.h"

Optimizer *optimizer_stochastic_gradient_descent_new(float learning_rate);

#endif
