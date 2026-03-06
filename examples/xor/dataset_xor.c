#include "dataset_xor.h"

#include <stdlib.h>

#define XOR_COMBINATIONS 4
#define XOR_INPUT_DIM 2
#define XOR_OUTPUT_DIM 2

typedef struct
{
    float inputs[XOR_COMBINATIONS][XOR_INPUT_DIM];
    size_t labels[XOR_COMBINATIONS];
    size_t order[XOR_COMBINATIONS]; // order used for indexing and shuffle
    size_t cursor;                  // pointer used as iterator
} DatasetXOR;

static size_t dataset_xor_size(Dataset *self)
{
    return XOR_COMBINATIONS;
}

static void dataset_xor_shuffle(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetXOR *dataset_xor = self->impl;

    size_t i = XOR_COMBINATIONS;
    while (i-- > 1)
    {
        size_t j = (size_t)(rand() % (i + 1));

        size_t aux = dataset_xor->order[i];
        dataset_xor->order[i] = dataset_xor->order[j];
        dataset_xor->order[j] = aux;
    }
}

static bool dataset_xor_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
    if (!self || !self->impl || !inputs || !labels || batch_size == 0)
    {
        return false;
    }

    DatasetXOR *dataset_xor = self->impl;
    if (dataset_xor->cursor >= XOR_COMBINATIONS)
    {
        return false;
    }

    size_t samples_left = XOR_COMBINATIONS - dataset_xor->cursor;
    size_t current_batch_size = batch_size < samples_left ? batch_size : samples_left;

    size_t input_shape[2] = {XOR_INPUT_DIM, current_batch_size};
    size_t label_shape[2] = {XOR_OUTPUT_DIM, current_batch_size};

    Tensor *batch_inputs = tensor_new(2, input_shape);
    Tensor *batch_labels = tensor_new(2, label_shape);
    if (!batch_inputs || !batch_labels)
    {
        tensor_free(batch_inputs);
        tensor_free(batch_labels);

        return false;
    }

    tensor_zero(batch_labels);

    for (size_t i = 0; i < current_batch_size; i++)
    {
        size_t sample_index = dataset_xor->order[dataset_xor->cursor + i];

        batch_inputs->data[i] = dataset_xor->inputs[sample_index][0];
        batch_inputs->data[current_batch_size + i] = dataset_xor->inputs[sample_index][1];

        size_t label_index = dataset_xor->labels[sample_index];
        batch_labels->data[label_index * current_batch_size + i] = 1;
    }

    dataset_xor->cursor += current_batch_size;

    *inputs = batch_inputs;
    *labels = batch_labels;

    return true;
}

static void dataset_xor_reset(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetXOR *dataset_xor = self->impl;
    dataset_xor->cursor = 0;

    for (size_t i = 0; i < XOR_COMBINATIONS; i++)
    {
        dataset_xor->order[i] = i;
    }
}

static void dataset_xor_free(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    free(self->impl);
}

Dataset *dataset_xor_new(void)
{
    DatasetXOR *dataset_xor = calloc(1, sizeof(DatasetXOR));
    if (!dataset_xor)
    {
        return NULL;
    }

    dataset_xor->inputs[0][0] = 0;
    dataset_xor->inputs[0][1] = 0;
    dataset_xor->inputs[1][0] = 0;
    dataset_xor->inputs[1][1] = 1;
    dataset_xor->inputs[2][0] = 1;
    dataset_xor->inputs[2][1] = 0;
    dataset_xor->inputs[3][0] = 1;
    dataset_xor->inputs[3][1] = 1;

    dataset_xor->labels[0] = 0;
    dataset_xor->labels[1] = 1;
    dataset_xor->labels[2] = 1;
    dataset_xor->labels[3] = 0;

    for (size_t i = 0; i < XOR_COMBINATIONS; i++)
    {
        dataset_xor->order[i] = i;
    }

    static const DatasetOps ops = {
        .size = dataset_xor_size,
        .shuffle = dataset_xor_shuffle,
        .next_batch = dataset_xor_next_batch,
        .reset = dataset_xor_reset,
        .free = dataset_xor_free};

    Dataset *dataset = dataset_new(dataset_xor, &ops);
    if (!dataset)
    {
        free(dataset_xor);
        return NULL;
    }

    return dataset;
}
