#include "dataset_engine.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENGINE_FEATURES 24
#define ENGINE_LABELS 1
#define ENGINE_COLUMNS 26

typedef struct
{
    float *inputs; // count * ENGINE_FEATURES
    float *labels; // count
    size_t *order; // count
    float feature_min[ENGINE_FEATURES];
    float feature_max[ENGINE_FEATURES];
    float max_rul;
    size_t count;
    size_t cursor;
} DatasetEngine;

static bool parse_engine_row(char *line, float values[ENGINE_COLUMNS])
{
    char *p = line;
    for (size_t i = 0; i < ENGINE_COLUMNS; i++)
    {
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }

        if (*p == '\0' || *p == '\n' || *p == '\r')
        {
            return false;
        }

        char *endptr = NULL;
        values[i] = strtof(p, &endptr);
        if (endptr == p)
        {
            return false;
        }
        p = endptr;
    }

    return true;
}

static size_t dataset_engine_size(Dataset *self)
{
    if (!self || !self->impl)
    {
        return 0;
    }

    DatasetEngine *dataset_engine = self->impl;

    return dataset_engine->count;
}

static void dataset_engine_shuffle(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEngine *dataset_engine = self->impl;

    size_t i = dataset_engine->count;
    while (i-- > 1)
    {
        size_t j = (size_t)(rand() % (i + 1));

        size_t aux = dataset_engine->order[i];
        dataset_engine->order[i] = dataset_engine->order[j];
        dataset_engine->order[j] = aux;
    }
}

static bool dataset_engine_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
    if (!self || !self->impl || !inputs || !labels || batch_size == 0)
    {
        return false;
    }

    DatasetEngine *dataset_engine = self->impl;

    if (dataset_engine->cursor >= dataset_engine->count)
    {
        return false;
    }

    size_t remaining = dataset_engine->count - dataset_engine->cursor;
    size_t current_batch_size = batch_size < remaining ? batch_size : remaining;

    size_t inputs_shape[2] = {ENGINE_FEATURES, current_batch_size};
    size_t labels_shape[2] = {ENGINE_LABELS, current_batch_size};

    Tensor *batch_inputs = tensor_new(2, inputs_shape);
    Tensor *batch_labels = tensor_new(2, labels_shape);
    if (!batch_inputs || !batch_labels)
    {
        tensor_free(batch_inputs);
        tensor_free(batch_labels);
        return false;
    }

    for (size_t i = 0; i < current_batch_size; i++)
    {
        size_t idx = dataset_engine->order[dataset_engine->cursor + i];
        size_t base = idx * ENGINE_FEATURES;

        for (size_t j = 0; j < ENGINE_FEATURES; j++)
        {
            float value = dataset_engine->inputs[base + j];
            float min_value = dataset_engine->feature_min[j];
            float max_value = dataset_engine->feature_max[j];
            float range = max_value - min_value;
            float scaled = range > 0.0f ? (value - min_value) / range : 0.0f;
            batch_inputs->data[i + j * current_batch_size] = scaled;
        }

        batch_labels->data[i] = dataset_engine->labels[idx];
    }

    dataset_engine->cursor += current_batch_size;
    *inputs = batch_inputs;
    *labels = batch_labels;

    return true;
}

static void dataset_engine_reset(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEngine *dataset_engine = self->impl;
    dataset_engine->cursor = 0;

    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        dataset_engine->order[i] = i;
    }
}

static void dataset_engine_free(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetEngine *dataset_engine = self->impl;
    free(dataset_engine->inputs);
    free(dataset_engine->labels);
    free(dataset_engine->order);
    free(self->impl);
}

Dataset *dataset_engine_new(void)
{
    DatasetEngine *dataset_engine = malloc(sizeof(DatasetEngine));
    if (!dataset_engine)
    {
        return NULL;
    }

    const char *dataset_path = "datasets/engine/train_FD001.txt";
    FILE *fptr = fopen(dataset_path, "r");
    if (!fptr)
    {
        free(dataset_engine);
        return NULL;
    }

    size_t capacity = 4096;
    dataset_engine->inputs = malloc(capacity * ENGINE_FEATURES * sizeof(float));
    dataset_engine->labels = malloc(capacity * sizeof(float));
    dataset_engine->order = NULL;
    dataset_engine->count = 0;
    dataset_engine->cursor = 0;
    dataset_engine->max_rul = 0;

    if (!dataset_engine->inputs || !dataset_engine->labels)
    {
        fclose(fptr);
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine);
        return NULL;
    }

    for (size_t j = 0; j < ENGINE_FEATURES; j++)
    {
        dataset_engine->feature_min[j] = FLT_MAX;
        dataset_engine->feature_max[j] = -FLT_MAX;
    }

    size_t engine_capacity = 4096;
    size_t *engine_ids = malloc(engine_capacity * sizeof(size_t));
    size_t *cycles = malloc(engine_capacity * sizeof(size_t));
    if (!engine_ids || !cycles)
    {
        fclose(fptr);
        free(engine_ids);
        free(cycles);
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine);
        return NULL;
    }

    char line[1024];
    while (fgets(line, sizeof(line), fptr))
    {
        float values[ENGINE_COLUMNS];
        if (!parse_engine_row(line, values))
        {
            continue;
        }

        size_t unit_id = (size_t)values[0];
        size_t cycle = (size_t)values[1];

        if (dataset_engine->count >= capacity)
        {
            capacity *= 2;
            float *new_inputs = realloc(dataset_engine->inputs, capacity * ENGINE_FEATURES * sizeof(float));
            float *new_labels = realloc(dataset_engine->labels, capacity * sizeof(float));
            if (!new_inputs || !new_labels)
            {
                free(new_inputs);
                free(new_labels);
                fclose(fptr);
                free(engine_ids);
                free(cycles);
                free(dataset_engine->inputs);
                free(dataset_engine->labels);
                free(dataset_engine);
                return NULL;
            }
            dataset_engine->inputs = new_inputs;
            dataset_engine->labels = new_labels;
        }

        if (dataset_engine->count >= engine_capacity)
        {
            engine_capacity *= 2;
            size_t *new_engine_ids = realloc(engine_ids, engine_capacity * sizeof(size_t));
            size_t *new_cycles = realloc(cycles, engine_capacity * sizeof(size_t));
            if (!new_engine_ids || !new_cycles)
            {
                free(new_engine_ids);
                free(new_cycles);
                fclose(fptr);
                free(engine_ids);
                free(cycles);
                free(dataset_engine->inputs);
                free(dataset_engine->labels);
                free(dataset_engine);
                return NULL;
            }
            engine_ids = new_engine_ids;
            cycles = new_cycles;
        }

        size_t base = dataset_engine->count * ENGINE_FEATURES;
        for (size_t j = 0; j < ENGINE_FEATURES; j++)
        {
            float value = values[j + 2];
            dataset_engine->inputs[base + j] = value;

            if (value < dataset_engine->feature_min[j])
            {
                dataset_engine->feature_min[j] = value;
            }
            if (value > dataset_engine->feature_max[j])
            {
                dataset_engine->feature_max[j] = value;
            }
        }

        engine_ids[dataset_engine->count] = unit_id;
        cycles[dataset_engine->count] = cycle;
        dataset_engine->labels[dataset_engine->count] = 0;

        dataset_engine->count++;
    }

    fclose(fptr);

    if (dataset_engine->count == 0)
    {
        free(engine_ids);
        free(cycles);
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine);
        return NULL;
    }

    size_t max_engine_id = 0;
    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        if (engine_ids[i] > max_engine_id)
        {
            max_engine_id = engine_ids[i];
        }
    }

    size_t *max_cycles = calloc(max_engine_id + 1, sizeof(size_t));
    if (!max_cycles)
    {
        free(engine_ids);
        free(cycles);
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine);
        return NULL;
    }

    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        size_t unit_id = engine_ids[i];
        if (cycles[i] > max_cycles[unit_id])
        {
            max_cycles[unit_id] = cycles[i];
        }
    }

    float max_rul = 0;
    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        size_t unit_id = engine_ids[i];
        size_t max_cycle = max_cycles[unit_id];
        float rul = (float)(max_cycle - cycles[i]);
        dataset_engine->labels[i] = rul;
        if (rul > max_rul)
        {
            max_rul = rul;
        }
    }

    dataset_engine->max_rul = max_rul > 0 ? max_rul : 1;

    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        dataset_engine->labels[i] = dataset_engine->labels[i] / dataset_engine->max_rul;
    }

    free(engine_ids);
    free(cycles);
    free(max_cycles);

    dataset_engine->order = malloc(dataset_engine->count * sizeof(size_t));
    if (!dataset_engine->order)
    {
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine);
        return NULL;
    }

    for (size_t i = 0; i < dataset_engine->count; i++)
    {
        dataset_engine->order[i] = i;
    }

    static const DatasetOps ops = {
        .size = dataset_engine_size,
        .shuffle = dataset_engine_shuffle,
        .next_batch = dataset_engine_next_batch,
        .reset = dataset_engine_reset,
        .free = dataset_engine_free};

    Dataset *dataset = dataset_new(dataset_engine, &ops);
    if (!dataset)
    {
        free(dataset_engine->inputs);
        free(dataset_engine->labels);
        free(dataset_engine->order);
        free(dataset_engine);
        return NULL;
    }

    return dataset;
}
