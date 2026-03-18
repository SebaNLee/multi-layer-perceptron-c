#include "dataset_meteorite.h"
#include "mlp/layers/dense.h"
#include "mlp/layers/relu.h"
#include "mlp/layers/softmax.h"
#include "mlp/layers/sigmoid.h"
#include "mlp/loss/cross_entropy.h"
#include "mlp/mlp.h"
#include "mlp/optimizer.h"
#include "mlp/optimizer/stochastic_gradient_descent.h"
#include <stdio.h>
#include <time.h>

#define EPOCHS 40
#define BATCH_SIZE 32
#define METEORITE_FEATURES 4
#define METEORITE_CLASSES 2

int main(int argc, char *argv[])
{
    clock_t start, end;
    double runtime;
    start = clock();

    MLP *mlp = mlp_new();
    mlp_set_seed(time(NULL));

    mlp_add_layer(mlp, layer_dense_new(METEORITE_FEATURES, 256, DENSE_INIT_HE));
    mlp_add_layer(mlp, layer_relu_new());
    mlp_add_layer(mlp, layer_dense_new(256, 256, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_sigmoid_new());
    mlp_add_layer(mlp, layer_dense_new(256, 256, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_sigmoid_new());
    mlp_add_layer(mlp, layer_dense_new(256, METEORITE_CLASSES, DENSE_INIT_XAVIER));
    mlp_add_layer(mlp, layer_softmax_new());

    Loss *loss = loss_cross_entropy_new();
    Optimizer *optimizer = optimizer_stochastic_gradient_descent_new(0.1);
    Dataset *dataset = dataset_meteorite_new();

    size_t total_size = dataset_size(dataset);
    size_t train_size = (size_t)((float)total_size * 0.8);
    size_t test_size = total_size - train_size;

    printf("\nTRAINING\n");
    printf("Epoch ./.\nAvg Loss:\n");

    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        float epoch_loss_sum = 0.0;
        size_t epoch_batches = 0;
        size_t train_count = 0;

        dataset_reset(dataset);

        while (train_count < train_size)
        {
            size_t remaining = train_size - train_count;
            size_t current_batch = remaining < BATCH_SIZE ? remaining : BATCH_SIZE;

            Tensor *input = NULL;
            Tensor *label = NULL;
            if (!dataset_next_batch(dataset, &input, &label, current_batch))
            {
                printf("Ok... this should not happen, DB was probably updated.\n");
                return 1;
            }

            Tensor *output = mlp_forward(mlp, input);
            epoch_loss_sum += loss_forward(loss, output, label);

            Tensor *gradient = loss_backward(loss, output, label);
            mlp_backward(mlp, gradient);
            optimizer_step(optimizer, mlp);

            tensor_free(gradient);
            tensor_free(input);
            tensor_free(label);

            epoch_batches++;
            train_count += current_batch;
        }

        if (epoch_batches > 0)
        {
            printf("\033[2A\033[2K\033[1B\033[2K\033[1A");
            printf("Epoch %ld/%d\n", epoch + 1, EPOCHS);
            printf("Avg Loss: %f\n", epoch_loss_sum / epoch_batches);
            fflush(stdout);
        }
    }

    dataset_reset(dataset);

    size_t skipped = 0;
    while (skipped < train_size)
    {
        size_t remaining = train_size - skipped;
        size_t current_batch = remaining < BATCH_SIZE ? remaining : BATCH_SIZE;

        Tensor *input = NULL;
        Tensor *label = NULL;
        if (!dataset_next_batch(dataset, &input, &label, current_batch))
        {
            break;
        }

        tensor_free(input);
        tensor_free(label);
        skipped += current_batch;
    }

    Tensor *input = NULL;
    Tensor *label = NULL;
    size_t correct = 0;
    size_t total = 0;
    float test_loss_sum = 0;
    size_t test_batches = 0;
    float average_true_class_confidence_sum = 0;

    while (dataset_next_batch(dataset, &input, &label, BATCH_SIZE))
    {
        Tensor *output = mlp_forward(mlp, input);
        test_loss_sum += loss_forward(loss, output, label);
        test_batches++;

        size_t current_batch_size = input->shape[1];

        for (size_t i = 0; i < current_batch_size; i++)
        {
            size_t prediction_class = 0;
            float prediction_value = output->data[i];

            for (size_t j = 0; j < METEORITE_CLASSES; j++)
            {
                size_t idx = j * current_batch_size + i;
                float current_prediction = output->data[idx];

                if (current_prediction > prediction_value)
                {
                    prediction_value = current_prediction;
                    prediction_class = j;
                }
            }

            size_t target_class = 0;
            float target_value = label->data[i];

            for (size_t j = 0; j < METEORITE_CLASSES; j++)
            {
                size_t idx = j * current_batch_size + i;
                float current_target = label->data[idx];

                if (current_target > target_value)
                {
                    target_value = current_target;
                    target_class = j;
                }
            }

            if (prediction_class == target_class)
            {
                correct++;
            }

            average_true_class_confidence_sum += output->data[target_class * current_batch_size + i];
            total++;
        }

        tensor_free(input);
        tensor_free(label);
    }

    if (total > 0)
    {
        float accuracy = ((float)correct / total) * 100;
        float average_true_class_confidence = (average_true_class_confidence_sum / total) * 100;
        printf("\nRESULTS\n");
        printf("Model Accuracy                : %9.5f%% (%ld/%ld)\n", accuracy, correct, total);
        printf("Average True Class Confidence : %9.5f%%\n", average_true_class_confidence);
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
