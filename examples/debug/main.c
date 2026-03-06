#include <stdio.h>

#include "mlp/layers/dense.h"
#include "mlp/layers/relu.h"
#include "mlp/layers/sigmoid.h"
#include "mlp/layers/softmax.h"
#include "mlp/loss/mean_squared_error.h"
#include "mlp/mlp.h"
#include "mlp/optimizer.h"
#include "mlp/optimizer/stochastic_gradient_descent.h"
#include "mlp/tensor.h"

int main(int argc, char *argv[])
{
    MLP *mlp = mlp_new();
    // mlp_set_seed(time(NULL));
    mlp_set_seed(22); // TODO hardcode seed before bathc impl

    mlp_add_layer(mlp, layer_dense_new(4, 8, DENSE_INIT_HE));
    mlp_add_layer(mlp, layer_relu_new());
    mlp_add_layer(mlp, layer_dense_new(8, 3, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_softmax_new());

    // debug input
    size_t shape[] = {4, 1};
    Tensor *input = tensor_new(2, shape);
    for (size_t i = 0; i < input->size; i++)
    {
        input->data[i] = i;
    }

    // debug output
    size_t y_shape[] = {3, 1};
    Tensor *y = tensor_new(2, y_shape);
    tensor_zero(y);
    y->data[1] = 1; // hardcode label

    Loss *loss = loss_mean_squared_error_new();
    Optimizer *optimizer = optimizer_stochastic_gradient_descent_new(0.1);

    for (size_t epoch = 0; epoch < 100; epoch++)
    {
        Tensor *output = mlp_forward(mlp, input);

        float loss_value = loss_forward(loss, output, y);
        Tensor *gradient = loss_backward(loss, output, y);

        mlp_backward(mlp, gradient);
        optimizer_step(optimizer, mlp);

        if (epoch % 10 == 0)
        {
            printf("Epoch: %ld\n", epoch);
            printf("Outputs:\n");
            float sum = 0;
            for (size_t i = 0; i < output->size; i++)
            {
                printf("[%ld]: %f\n", i, output->data[i]);
                sum += output->data[i];
            }
            printf("Softmax sum: %f\n", sum);
            printf("Loss value: %f\n", loss_value);
            printf("\n");
        }

        tensor_free(gradient);
    }

    loss_free(loss);
    tensor_free(y);
    tensor_free(input);
    optimizer_free(optimizer);
    mlp_free(mlp);

    return 0;
}