#include <stddef.h>
#include "layer.h"
#include "tensor.h"

typedef struct
{
    Layer **layers;
    size_t layers_count;
} MLP;

// TODO
MLP *mlp_new();
void mlp_add_layer(MLP *mlp, Layer *l);
Tensor *mlp_forward(MLP *mlp);
void mlp_backward(MLP *mlp);
void mlp_apply_gradients(MLP *mlp);