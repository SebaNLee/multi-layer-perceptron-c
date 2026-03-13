#include "dataset_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    // TODO
} DatasetEngine;

static size_t dataset_engine_size(Dataset *self)
{
}

static void dataset_engine_shuffle(Dataset *self)
{
}

static bool dataset_engine_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
}

static void dataset_engine_reset(Dataset *self)
{
}

static void dataset_engine_free(Dataset *self)
{
}

Dataset *dataset_engine_new(void)
{
    DatasetEngine *dataset_engine = malloc(sizeof(DatasetEngine));
    if (!dataset_engine)
    {
        return NULL;
    }

    const char *dataset_path = "datasets/meteorite/Meteorite_Landings.csv";
    FILE *fptr = fopen(dataset_path, "r");
    if (!fptr)
    {
        free(dataset_engine);
        return NULL;
    }

    // TODO

    static const DatasetOps ops = {
        .size = dataset_engine_size,
        .shuffle = dataset_engine_shuffle,
        .next_batch = dataset_engine_next_batch,
        .reset = dataset_engine_reset,
        .free = dataset_engine_free};

    Dataset *dataset = dataset_new(dataset_engine, &ops);
    if (!dataset)
    {
        // free(dataset_engine->inputs);
        // free(dataset_engine->labels);
        // free(dataset_engine->order);
        free(dataset_engine);
        return NULL;
    }

    return dataset;
}
