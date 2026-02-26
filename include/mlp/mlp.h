#include "layer.h"
#include "tensor.h"
#include <stddef.h>

typedef struct
{
    Layer **layers;
    size_t layers_count;
} MLP;

MLP *mlp_new();
void mlp_free(MLP *mlp);
void mlp_add_layer(MLP *mlp, Layer *layer);

Tensor *mlp_forward(MLP *mlp, Tensor *input);
void mlp_backward(MLP *mlp, Tensor *gradient_output);
