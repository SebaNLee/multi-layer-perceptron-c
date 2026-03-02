#include "mlp/layers/dense.h"
#include "mlp/layers/relu.h"
#include "mlp/layers/sigmoid.h"
#include "mlp/layers/softmax.h"
#include "mlp/loss/cross_entropy.h"
#include "mlp/mlp.h"
#include "mlp/optimizer.h"
#include "mlp/optimizer/stochastic_gradient_descent.h"
#include "mlp/tensor.h"
#include <stdio.h>
#include <time.h>

#define EPOCHS 5000

int main(int argc, char *argv[])
{
    MLP *mlp = mlp_new();
    mlp_set_seed(time(NULL));

    mlp_add_layer(mlp, layer_dense_new(2, 8, DENSE_INIT_HE));
    mlp_add_layer(mlp, layer_relu_new());
    mlp_add_layer(mlp, layer_dense_new(8, 2, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_softmax_new());

    Loss *loss = loss_cross_entropy_new();
    Optimizer *optimizer = optimizer_stochastic_gradient_descent_new(0.1);

    size_t input_shape[] = {2, 1};
    size_t label_shape[] = {2, 1};

    Tensor *input = tensor_new(2, input_shape);
    Tensor *label = tensor_new(2, label_shape);

    float xor_data[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    int xor_label[4] = {0, 1, 1, 0};

    printf("\n===TRAINING===\n");

    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        float loss_value = 0;

        for (size_t i = 0; i < 4; i++)
        {
            input->data[0] = xor_data[i][0];
            input->data[1] = xor_data[i][1];
            tensor_zero(label);
            label->data[xor_label[i]] = 1;

            Tensor *output = mlp_forward(mlp, input);
            loss_value += loss_forward(loss, output, label);

            Tensor *gradient = loss_backward(loss, output, label);
            mlp_backward(mlp, gradient);
            optimizer_step(optimizer, mlp);

            tensor_free(gradient);
        }

        if (epoch % 500 == 0)
        {
            printf("Epoch: %ld\n", epoch);
            printf("Loss: %f\n", loss_value);
            printf("\n");
        }
    }

    printf("\n===RESULTS===\n");

    for (size_t i = 0; i < 4; i++)
    {
        input->data[0] = xor_data[i][0];
        input->data[1] = xor_data[i][1];
        tensor_zero(label);
        label->data[xor_label[i]] = 1;

        Tensor *output = mlp_forward(mlp, input);
        float loss_value = loss_forward(loss, output, label);

        printf("%f XOR %f → [%f, %f]\n", xor_data[i][0], xor_data[i][1], output->data[0], output->data[1]);
        printf("Loss: %f\n", loss_value);
    }

    loss_free(loss);
    tensor_free(input);
    tensor_free(label);
    optimizer_free(optimizer);
    mlp_free(mlp);

    return 0;
}