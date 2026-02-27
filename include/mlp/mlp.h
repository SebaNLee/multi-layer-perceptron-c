#ifndef MLP_MLP_H
#define MLP_LOSS_H

#include "mlp/layer.h"
#include "mlp/tensor.h"
#include <stddef.h>

#define BLOCK 5

typedef struct
{
    Layer **layers;
    size_t layers_count;
    size_t layers_size;
} MLP;

MLP *mlp_new();
void mlp_free(MLP *mlp);
void mlp_add_layer(MLP *mlp, Layer *layer);

Tensor *mlp_forward(MLP *mlp, Tensor *input);
void mlp_backward(MLP *mlp, Tensor *gradient_output);

#endif