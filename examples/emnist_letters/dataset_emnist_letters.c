#include "dataset_emnist_letters.h"

#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define EMNIST_LETTERS_CLASSES 26

typedef struct
{
    uint8_t *images; // n * 28 * 28
    uint8_t *labels; // n
    size_t *order;   // n
    size_t count;
    size_t cursor;
    size_t rows;
    size_t cols;
} DatasetEmnistLetters;

static uint32_t read_u32_bigendian(FILE *fptr)
{
    uint32_t raw = 0;
    if (fread(&raw, sizeof(raw), 1, fptr) != 1)
    {
        return 0;
    }

    return ntohl(raw);
}

static size_t dataset_emnist_letters_size(Dataset *self)
{
    if (!self || !self->impl)
    {
        return 0;
    }

    DatasetEmnistLetters *dataset_emnist_letters = self->impl;

    return dataset_emnist_letters->count;
}

static void dataset_emnist_letters_shuffle(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEmnistLetters *dataset_emnist_letters = self->impl;

    size_t i = dataset_emnist_letters->count;
    while (i-- > 1)
    {
        size_t j = (size_t)(rand() % (i + 1));

        size_t aux = dataset_emnist_letters->order[i];
        dataset_emnist_letters->order[i] = dataset_emnist_letters->order[j];
        dataset_emnist_letters->order[j] = aux;
    }
}

static bool dataset_emnist_letters_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
    if (!self || !self->impl || !inputs || !labels || batch_size == 0)
    {
        return false;
    }

    DatasetEmnistLetters *dataset_emnist_letters = self->impl;

    if (dataset_emnist_letters->cursor >= dataset_emnist_letters->count)
    {
        return false;
    }

    size_t remaining = dataset_emnist_letters->count - dataset_emnist_letters->cursor;
    size_t current_batch_size = batch_size < remaining ? batch_size : remaining;

    size_t inputs_shape[2] = {dataset_emnist_letters->rows * dataset_emnist_letters->cols, current_batch_size};
    size_t labels_shape[2] = {EMNIST_LETTERS_CLASSES, current_batch_size};

    Tensor *batch_inputs = tensor_new(2, inputs_shape);
    Tensor *batch_labels = tensor_new(2, labels_shape);
    if (!batch_inputs || !batch_labels)
    {
        tensor_free(batch_inputs);
        tensor_free(batch_labels);
        return false;
    }

    tensor_zero(batch_labels);

    for (size_t i = 0; i < current_batch_size; i++)
    {
        size_t idx = dataset_emnist_letters->order[dataset_emnist_letters->cursor + i];
        size_t base = idx * dataset_emnist_letters->rows * dataset_emnist_letters->cols;

        for (size_t j = 0; j < dataset_emnist_letters->rows * dataset_emnist_letters->cols; j++)
        {
            batch_inputs->data[i + j * current_batch_size] = dataset_emnist_letters->images[base + j] / (float)255;
        }

        uint8_t class = dataset_emnist_letters->labels[idx];
        size_t class_index = (size_t) class - 1;
        batch_labels->data[class_index * current_batch_size + i] = 1;
    }

    dataset_emnist_letters->cursor += current_batch_size;
    *inputs = batch_inputs;
    *labels = batch_labels;

    return true;
}

static void dataset_emnist_letters_reset(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEmnistLetters *dataset_emnist_letters = self->impl;

    dataset_emnist_letters->cursor = 0;

    for (size_t i = 0; i < dataset_emnist_letters->count; i++)
    {
        dataset_emnist_letters->order[i] = i;
    }
}

static void dataset_emnist_letters_free(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEmnistLetters *dataset_emnist_letters = self->impl;

    free(dataset_emnist_letters->images);
    free(dataset_emnist_letters->labels);
    free(dataset_emnist_letters->order);
    free(self->impl);
}

Dataset *dataset_emnist_letters_new(EmnistSubset subset)
{
    DatasetEmnistLetters *dataset_emnist_letters = malloc(sizeof(DatasetEmnistLetters));
    if (!dataset_emnist_letters)
    {
        return NULL;
    }

    // emnist letters data load from /datasets/emnist
    char *images_path = NULL;
    char *labels_path = NULL;
    if (subset == TRAIN)
    {
        images_path = "datasets/emnist/emnist-letters-train-images-idx3-ubyte";
        labels_path = "datasets/emnist/emnist-letters-train-labels-idx1-ubyte";
    }
    else if (subset == TEST)
    {
        images_path = "datasets/emnist/emnist-letters-test-images-idx3-ubyte";
        labels_path = "datasets/emnist/emnist-letters-test-labels-idx1-ubyte";
    }
    else
    {
        free(dataset_emnist_letters);
        return NULL;
    }

    FILE *images_fptr = fopen(images_path, "rb");
    FILE *labels_fptr = fopen(labels_path, "rb");
    if (!images_fptr || !labels_fptr)
    {
        if (images_fptr)
        {
            fclose(images_fptr);
        }

        if (labels_fptr)
        {
            fclose(labels_fptr);
        }

        free(dataset_emnist_letters);
        return NULL;
    }

    // format:
    // http://yann.lecun.com/exdb/mnist/
    // https://github.com/afrozenator/mnist-parser

    // images header
    uint32_t images_magic_number = read_u32_bigendian(images_fptr);
    uint32_t image_count = read_u32_bigendian(images_fptr);
    uint32_t image_rows = read_u32_bigendian(images_fptr);
    uint32_t image_cols = read_u32_bigendian(images_fptr);

    // labels
    uint32_t labels_magic_number = read_u32_bigendian(labels_fptr);
    uint32_t labels_count = read_u32_bigendian(labels_fptr);

    dataset_emnist_letters->count = (size_t)image_count;
    dataset_emnist_letters->rows = (size_t)image_rows;
    dataset_emnist_letters->cols = (size_t)image_cols;
    dataset_emnist_letters->cursor = 0;
    dataset_emnist_letters->images = malloc(image_count * image_rows * image_cols);
    dataset_emnist_letters->labels = malloc(labels_count);
    dataset_emnist_letters->order = malloc(labels_count * sizeof(size_t));
    if (!dataset_emnist_letters->images || !dataset_emnist_letters->labels || !dataset_emnist_letters->order)
    {
        fclose(images_fptr);
        fclose(labels_fptr);
        free(dataset_emnist_letters->images);
        free(dataset_emnist_letters->labels);
        free(dataset_emnist_letters->order);
        free(dataset_emnist_letters);
        return NULL;
    }

    // read images and labels rows
    if (fread(dataset_emnist_letters->images, 1, image_count * image_rows * image_cols, images_fptr) != image_count * image_rows * image_cols ||
        fread(dataset_emnist_letters->labels, 1, labels_count, labels_fptr) != labels_count)
    {
        fclose(images_fptr);
        fclose(labels_fptr);
        free(dataset_emnist_letters->images);
        free(dataset_emnist_letters->labels);
        free(dataset_emnist_letters->order);
        free(dataset_emnist_letters);
        return NULL;
    }

    fclose(images_fptr);
    fclose(labels_fptr);

    for (size_t i = 0; i < dataset_emnist_letters->count; i++)
    {
        dataset_emnist_letters->order[i] = i;
    }

    static const DatasetOps ops = {
        .size = dataset_emnist_letters_size,
        .shuffle = dataset_emnist_letters_shuffle,
        .next_batch = dataset_emnist_letters_next_batch,
        .reset = dataset_emnist_letters_reset,
        .free = dataset_emnist_letters_free};

    Dataset *dataset = dataset_new(dataset_emnist_letters, &ops);
    if (!dataset)
    {
        free(dataset_emnist_letters->images);
        free(dataset_emnist_letters->labels);
        free(dataset_emnist_letters->order);
        free(dataset_emnist_letters);
        return NULL;
    }

    return dataset;
}
