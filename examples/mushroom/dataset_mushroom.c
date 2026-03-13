#include "dataset_mushroom.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT 8124
#define MUSHROOM_FEATURES 127
#define MUSHROOM_LABELS 2
#define DATASET_COLUMNS 23

typedef struct
{
    float *inputs;  // count * MUSHROOM_FEATURES
    size_t *labels; // count
    size_t *order;  // count
    size_t cursor;
} DatasetMushroom;

static size_t dataset_mushroom_size(Dataset *self)
{
}

static void dataset_mushroom_shuffle(Dataset *self)
{
}

static bool dataset_mushroom_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
}

static void dataset_mushroom_reset(Dataset *self)
{
}

static void dataset_mushroom_free(Dataset *self)
{
}

Dataset *dataset_mushroom_new(void)
{
    // implementation: one-hot encoding for each

    DatasetMushroom *dataset_mushroom = malloc(sizeof(DatasetMushroom));
    if (!dataset_mushroom)
    {
        return NULL;
    }

    const char *dataset_path = "datasets/mushroom/agaricus-lepiota.data";
    FILE *fptr = fopen(dataset_path, "r");
    if (!fptr)
    {
        free(dataset_mushroom);
        return NULL;
    }

    dataset_mushroom->inputs = malloc(COUNT * MUSHROOM_FEATURES * sizeof(float));
    dataset_mushroom->labels = malloc(COUNT * sizeof(size_t));
    dataset_mushroom->order = NULL;
    dataset_mushroom->cursor = 0;

    if (!dataset_mushroom->inputs || !dataset_mushroom->labels)
    {
        fclose(fptr);
        free(dataset_mushroom->inputs);
        free(dataset_mushroom->labels);
        free(dataset_mushroom);
        return NULL;
    }

    memset(dataset_mushroom->inputs, 0, COUNT * MUSHROOM_FEATURES * sizeof(float));
    memset(dataset_mushroom->labels, 0, COUNT * sizeof(size_t));

    static const char *const column_variables[DATASET_COLUMNS] = {
        "pe",
        "bcxfks",
        "fgys",
        "nbygpruew",
        "tf",
        "alcyfmnps",
        "adfn",
        "cwd",
        "bn",
        "knbhgropuewy",
        "et",
        "bcuezr?",
        "fyks",
        "fyks",
        "nbcgopewy",
        "nbcgopewy",
        "pu",
        "nowy",
        "not",
        "ceflnpzs",
        "knbhrouwy",
        "acnpsvy",
        "glmpuwdy"};

    static int8_t labels_index[256];
    for (size_t i = 0; i < 256; i++)
    {
        labels_index[i] = -1;
    }
    for (size_t i = 0; column_variables[0][i]; i++)
    {
        unsigned char symbol = (unsigned char)column_variables[0][i];
        labels_index[symbol] = (int8_t)i;
    }

    static int8_t inputs_index[DATASET_COLUMNS][256];
    for (size_t column = 0; column < DATASET_COLUMNS; column++)
    {
        for (size_t i = 0; i < 256; i++)
        {
            inputs_index[column][i] = -1;
        }
    }
    size_t offset = 0;
    for (size_t column = 1; column < DATASET_COLUMNS; column++)
    {
        for (size_t i = 0; column_variables[column][i]; i++)
        {
            unsigned char symbol = (unsigned char)column_variables[column][i];
            inputs_index[column][symbol] = offset++;
        }
    }

    char line[1024];
    size_t row = 0;
    while (fgets(line, sizeof(line), fptr))
    {
        char *fields[DATASET_COLUMNS];
        char *token = strtok(line, ",");
        size_t i = 0;

        while (token && i < DATASET_COLUMNS)
        {
            fields[i++] = token;
            token = strtok(NULL, ",");
        }

        size_t *curr_labels = dataset_mushroom->labels + row;
        curr_labels[labels_index[(unsigned char)fields[0][0]]] = 1;

        float *curr_inputs = dataset_mushroom->inputs + row * MUSHROOM_FEATURES;
        for (size_t i = 1; i < DATASET_COLUMNS; i++)
        {
            curr_inputs[inputs_index[i][(unsigned char)fields[i][0]]] = 1;
        }

        row++;
    }

    fclose(fptr);

    dataset_mushroom->order = malloc(COUNT * sizeof(size_t));
    if (!dataset_mushroom->order)
    {
        free(dataset_mushroom->inputs);
        free(dataset_mushroom->labels);
        free(dataset_mushroom);
        return NULL;
    }

    for (size_t i = 0; i < COUNT; i++)
    {
        dataset_mushroom->order[i] = i;
    }

    static const DatasetOps ops = {
        .size = dataset_mushroom_size,
        .shuffle = dataset_mushroom_shuffle,
        .next_batch = dataset_mushroom_next_batch,
        .reset = dataset_mushroom_reset,
        .free = dataset_mushroom_free};

    Dataset *dataset = dataset_new(dataset_mushroom, &ops);
    if (!dataset)
    {
        free(dataset_mushroom->inputs);
        free(dataset_mushroom->labels);
        free(dataset_mushroom->order);
        free(dataset_mushroom);
        return NULL;
    }

    return dataset;
}
