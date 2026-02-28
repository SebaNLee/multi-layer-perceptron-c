#ifndef MLP_LAYERS_DENSE_H
#define MLP_LAYERS_DENSE_H

#include "mlp/layer.h"

typedef enum
{
    DENSE_INIT_XAVIER,
    DENSE_INIT_HE
} DenseInit;

Layer *layer_dense_new(size_t input, size_t output, DenseInit init);

#endif