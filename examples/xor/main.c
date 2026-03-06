#include "dataset_xor.h"
#include "mlp/layers/dense.h"
#include "mlp/layers/relu.h"
#include "mlp/layers/sigmoid.h"
#include "mlp/layers/softmax.h"
#include "mlp/loss/cross_entropy.h"
#include "mlp/mlp.h"
#include "mlp/optimizer.h"
#include "mlp/optimizer/stochastic_gradient_descent.h"
#include <stdio.h>
#include <time.h>

#define EPOCHS 5000
#define BATCH_SIZE 4

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
    Dataset *dataset = dataset_xor_new();

    printf("===TRAINING===\n");

    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        float loss_value = 0;
        size_t batch_count = 0;

        dataset_reset(dataset);
        dataset_shuffle(dataset);

        Tensor *input = NULL;
        Tensor *label = NULL;

        while (dataset_next_batch(dataset, &input, &label, BATCH_SIZE))
        {
            Tensor *output = mlp_forward(mlp, input);
            loss_value += loss_forward(loss, output, label);

            Tensor *gradient = loss_backward(loss, output, label);
            mlp_backward(mlp, gradient);
            optimizer_step(optimizer, mlp);

            tensor_free(gradient);
            tensor_free(input);
            tensor_free(label);

            batch_count++;
        }

        if (epoch % 500 == 0)
        {
            printf("Epoch: %ld\n", epoch);
            printf("Loss: %f\n", loss_value / batch_count);
            printf("\n");
        }
    }

    printf("===RESULTS===\n");

    dataset_reset(dataset);

    Tensor *input = NULL;
    Tensor *label = NULL;
    size_t correct = 0;
    size_t total = 0;
    float mean_confidence_sum = 0;

    while (dataset_next_batch(dataset, &input, &label, 1))
    {
        Tensor *output = mlp_forward(mlp, input);
        float loss_value = loss_forward(loss, output, label);

        size_t predicted_class = output->data[0] > output->data[1] ? 0 : 1;
        size_t target_class = label->data[0] > label->data[1] ? 0 : 1;
        if (predicted_class == target_class)
        {
            correct++;
        }
        mean_confidence_sum += output->data[target_class];
        total++;

        printf("%f XOR %f → [%f, %f]\n", input->data[0], input->data[1], output->data[0], output->data[1]);
        printf("Loss: %f\n", loss_value);

        tensor_free(input);
        tensor_free(label);
    }

    if (total > 0)
    {
        float accuracy = ((float)correct / total) * 100;
        float mean_confidence_prob = (mean_confidence_sum / total) * 100;
        printf("\n");
        printf("Model accuracy: %.2f%% (%ld/%ld)\n", accuracy, correct, total);
        printf("Mean confidence: %f%%\n", mean_confidence_prob);
    }

    dataset_free(dataset);
    loss_free(loss);
    optimizer_free(optimizer);
    mlp_free(mlp);

    return 0;
}
