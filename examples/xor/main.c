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
    clock_t start, end;
    double runtime;
    start = clock();

    MLP *mlp = mlp_new();
    mlp_set_seed(time(NULL));

    mlp_add_layer(mlp, layer_dense_new(2, 8, DENSE_INIT_HE));
    mlp_add_layer(mlp, layer_relu_new());
    mlp_add_layer(mlp, layer_dense_new(8, 2, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_softmax_new());

    Loss *loss = loss_cross_entropy_new();
    Optimizer *optimizer = optimizer_stochastic_gradient_descent_new(0.1);
    Dataset *dataset = dataset_xor_new();
    size_t total_size = dataset_size(dataset);
    size_t test_size = total_size;

    printf("\nTraining:\n");
    printf("Epoch ./.\nAvg Loss:\n");

    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        float epoch_loss_sum = 0;
        size_t epoch_batches = 0;

        dataset_reset(dataset);
        dataset_shuffle(dataset);

        Tensor *input = NULL;
        Tensor *label = NULL;

        while (dataset_next_batch(dataset, &input, &label, BATCH_SIZE))
        {
            Tensor *output = mlp_forward(mlp, input);
            epoch_loss_sum += loss_forward(loss, output, label);

            Tensor *gradient = loss_backward(loss, output, label);
            mlp_backward(mlp, gradient);
            optimizer_step(optimizer, mlp);

            tensor_free(gradient);
            tensor_free(input);
            tensor_free(label);

            epoch_batches++;
        }

        if ((((epoch + 1) % 500) == 0 || (epoch + 1) == EPOCHS) && epoch_batches > 0)
        {
            printf("\033[2A\033[2K\033[1B\033[2K\033[1A");
            printf("Epoch %ld/%d\n", epoch + 1, EPOCHS);
            printf("Avg Loss: %f\n", epoch_loss_sum / epoch_batches);
            fflush(stdout);
        }
    }

    dataset_reset(dataset);

    Tensor *input = NULL;
    Tensor *label = NULL;
    size_t correct = 0;
    size_t total = 0;
    float test_loss_sum = 0;
    size_t test_batches = 0;
    float mean_confidence_sum = 0;

    while (dataset_next_batch(dataset, &input, &label, 1))
    {
        Tensor *output = mlp_forward(mlp, input);
        float loss_value = loss_forward(loss, output, label);
        test_loss_sum += loss_value;
        test_batches++;

        size_t predicted_class = output->data[0] > output->data[1] ? 0 : 1;
        size_t target_class = label->data[0] > label->data[1] ? 0 : 1;
        if (predicted_class == target_class)
        {
            correct++;
        }
        mean_confidence_sum += output->data[target_class];
        total++;

        tensor_free(input);
        tensor_free(label);
    }

    if (total > 0)
    {
        float accuracy = ((float)correct / total) * 100;
        float mean_confidence_prob = (mean_confidence_sum / total) * 100;
        printf("\nResults:\n");
        printf("Model Accuracy                : %9.5f%% (%ld/%ld)\n", accuracy, correct, total);
        printf("Average True Class Confidence : %9.5f%%\n", mean_confidence_prob);
        printf("Mean Test Loss                : %9.5f\n", test_loss_sum / (float)test_batches);
        printf("Test Samples                  : %ld/%ld\n", test_size, total_size);
        printf("\n");
    }

    dataset_free(dataset);
    loss_free(loss);
    optimizer_free(optimizer);
    mlp_free(mlp);

    end = clock();
    runtime = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Runtime: %fs\n", runtime);

    return 0;
}
