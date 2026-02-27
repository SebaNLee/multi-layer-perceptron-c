#include <stdio.h>

#include "mlp/layers/dense.h"
#include "mlp/layers/relu.h"
#include "mlp/layers/sigmoid.h"
#include "mlp/layers/softmax.h"
#include "mlp/mlp.h"
#include "mlp/tensor.h"

int main(int argc, char *argv[])
{
    MLP *mlp = mlp_new();

    mlp_add_layer(mlp, layer_dense_new(4, 8));
    mlp_add_layer(mlp, layer_relu_new());
    mlp_add_layer(mlp, layer_dense_new(8, 3));
    mlp_add_layer(mlp, layer_softmax_new());

    // debug input
    size_t shape[] = {4, 1};
    Tensor *input = tensor_new(2, shape);
    for (size_t i = 0; i < input->size; i++)
    {
        input->data[i] = i;
    }

    Tensor *output = mlp_forward(mlp, input);

    float sum = 0;
    printf("Outputs:\n");
    for (size_t i = 0; i < output->size; i++)
    {
        printf("[%ld]: %f\n", i, output->data[i]);
        sum += output->data[i];
    }
    printf("Softmax sum: %f\n", sum);

    tensor_free(input);
    mlp_free(mlp);

    return 0;
}