#include "dataset_meteorite.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define METEORITE_FEATURES 4
#define METEORITE_LABELS 2
#define YEAR_SCALE 2100.0f
#define MASS_SCALE 1000000.0f
#define LAT_SCALE 90.0f
#define LON_SCALE 180.0f

typedef struct
{
    float *inputs;   // count * METEORITE_FEATURES
    uint8_t *labels; // count
    size_t *order;   // count
    size_t count;
    size_t cursor;
} DatasetMeteorite;

static bool parse_meteorite_row(char *line, float *year, float *mass, float *latitude, float *longitude, uint8_t *label)
{
    char *fields[9] = {0};
    size_t count = 0;
    char *p = line;

    while (*p && count < 9)
    {
        fields[count++] = p;
        while (*p && *p != ',' && *p != '\n' && *p != '\r')
        {
            p++;
        }

        *p = '\0';
        p++;
    }

    if (count < 9)
    {
        return false;
    }

    const char *mass_field = fields[4];
    const char *fall_field = fields[5];
    const char *year_field = fields[6];
    const char *latitude_field = fields[7];
    const char *longitude_field = fields[8];

    if (!mass_field || !*mass_field || !fall_field || !*fall_field || !year_field || !*year_field || !latitude_field || !*latitude_field || !longitude_field || !*longitude_field)
    {
        return false;
    }

    char *endptr = NULL;
    *year = strtof(year_field, &endptr);
    if (endptr == year_field)
    {
        return false;
    }

    *mass = strtof(mass_field, &endptr);
    if (endptr == mass_field)
    {
        return false;
    }

    *latitude = strtof(latitude_field, &endptr);
    if (endptr == latitude_field)
    {
        return false;
    }

    *longitude = strtof(longitude_field, &endptr);
    if (endptr == longitude_field)
    {
        return false;
    }

    if (strcmp(fall_field, "Fell") == 0)
    {
        *label = 1;
    }
    else if (strcmp(fall_field, "Found") == 0)
    {
        *label = 0;
    }
    else
    {
        return false;
    }

    return true;
}

static size_t dataset_meteorite_size(Dataset *self)
{
    if (!self || !self->impl)
    {
        return 0;
    }

    DatasetMeteorite *dataset_meteorite = self->impl;

    return dataset_meteorite->count;
}

static void dataset_meteorite_shuffle(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetMeteorite *dataset_meteorite = self->impl;

    size_t i = dataset_meteorite->count;
    while (i-- > 1)
    {
        size_t j = (size_t)(rand() % (i + 1));

        size_t aux = dataset_meteorite->order[i];
        dataset_meteorite->order[i] = dataset_meteorite->order[j];
        dataset_meteorite->order[j] = aux;
    }
}

static bool dataset_meteorite_next_batch(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size)
{
    if (!self || !self->impl || !inputs || !labels || batch_size == 0)
    {
        return false;
    }

    DatasetMeteorite *dataset_meteorite = self->impl;

    if (dataset_meteorite->cursor >= dataset_meteorite->count)
    {
        return false;
    }

    size_t remaining = dataset_meteorite->count - dataset_meteorite->cursor;
    size_t current_batch_size = batch_size < remaining ? batch_size : remaining;

    size_t inputs_shape[2] = {METEORITE_FEATURES, current_batch_size};
    size_t labels_shape[2] = {METEORITE_LABELS, current_batch_size};

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
        size_t idx = dataset_meteorite->order[dataset_meteorite->cursor + i];
        size_t base = idx * METEORITE_FEATURES;

        for (size_t j = 0; j < METEORITE_FEATURES; j++)
        {
            float value = dataset_meteorite->inputs[base + j];
            float scaled = value;
            if (j == 0)
            {
                scaled = value / YEAR_SCALE;
            }
            else if (j == 1)
            {
                scaled = value / MASS_SCALE;
            }
            else if (j == 2)
            {
                scaled = value / LAT_SCALE;
            }
            else if (j == 3)
            {
                scaled = value / LON_SCALE;
            }
            batch_inputs->data[i + j * current_batch_size] = scaled;
        }

        uint8_t label = dataset_meteorite->labels[idx];
        batch_labels->data[(size_t)label * current_batch_size + i] = 1.0f;
    }

    dataset_meteorite->cursor += current_batch_size;
    *inputs = batch_inputs;
    *labels = batch_labels;

    return true;
}

static void dataset_meteorite_reset(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetMeteorite *dataset_meteorite = self->impl;
    dataset_meteorite->cursor = 0;

    for (size_t i = 0; i < dataset_meteorite->count; i++)
    {
        dataset_meteorite->order[i] = i;
    }
}

static void dataset_meteorite_free(Dataset *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    DatasetMeteorite *dataset_meteorite = self->impl;
    free(dataset_meteorite->inputs);
    free(dataset_meteorite->labels);
    free(dataset_meteorite->order);
    free(self->impl);
}

Dataset *dataset_meteorite_new(void)
{
    DatasetMeteorite *dataset_meteorite = malloc(sizeof(DatasetMeteorite));
    if (!dataset_meteorite)
    {
        return NULL;
    }

    const char *dataset_path = "datasets/meteorite/Meteorite_Landings.csv";
    FILE *fptr = fopen(dataset_path, "r");
    if (!fptr)
    {
        free(dataset_meteorite);
        return NULL;
    }

    size_t capacity = 1024;
    dataset_meteorite->inputs = malloc(capacity * METEORITE_FEATURES * sizeof(float));
    dataset_meteorite->labels = malloc(capacity * sizeof(uint8_t));
    dataset_meteorite->order = NULL;
    dataset_meteorite->count = 0;
    dataset_meteorite->cursor = 0;

    if (!dataset_meteorite->inputs || !dataset_meteorite->labels)
    {
        fclose(fptr);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    char line[1024];
    if (!fgets(line, sizeof(line), fptr))
    {
        fclose(fptr);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    while (fgets(line, sizeof(line), fptr))
    {
        float year = 0.0f;
        float mass = 0.0f;
        float lat = 0.0f;
        float lon = 0.0f;
        uint8_t label = 0;
        if (!parse_meteorite_row(line, &year, &mass, &lat, &lon, &label))
        {
            continue;
        }

        if (dataset_meteorite->count >= capacity)
        {
            capacity *= 2;
            float *new_inputs = realloc(dataset_meteorite->inputs, capacity * METEORITE_FEATURES * sizeof(float));
            uint8_t *new_labels = realloc(dataset_meteorite->labels, capacity * sizeof(uint8_t));
            if (!new_inputs || !new_labels)
            {
                free(new_inputs);
                free(new_labels);
                fclose(fptr);
                free(dataset_meteorite->inputs);
                free(dataset_meteorite->labels);
                free(dataset_meteorite);
                return NULL;
            }
            dataset_meteorite->inputs = new_inputs;
            dataset_meteorite->labels = new_labels;
        }

        size_t base = dataset_meteorite->count * METEORITE_FEATURES;
        dataset_meteorite->inputs[base] = year;
        dataset_meteorite->inputs[base + 1] = mass;
        dataset_meteorite->inputs[base + 2] = lat;
        dataset_meteorite->inputs[base + 3] = lon;
        dataset_meteorite->labels[dataset_meteorite->count] = label;

        dataset_meteorite->count++;
    }

    fclose(fptr);

    if (dataset_meteorite->count == 0)
    {
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    size_t *fell_indices = malloc(dataset_meteorite->count * sizeof(size_t));
    size_t *found_indices = malloc(dataset_meteorite->count * sizeof(size_t));
    if (!fell_indices || !found_indices)
    {
        free(fell_indices);
        free(found_indices);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    size_t fell_count = 0;
    size_t found_count = 0;
    for (size_t i = 0; i < dataset_meteorite->count; i++)
    {
        if (dataset_meteorite->labels[i] == 1)
        {
            fell_indices[fell_count++] = i;
        }
        else
        {
            found_indices[found_count++] = i;
        }
    }

    // !! implementation: undersampling
    size_t min_count = fell_count < found_count ? fell_count : found_count;
    if (min_count == 0)
    {
        free(fell_indices);
        free(found_indices);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    size_t balanced_count = 2 * min_count;
    dataset_meteorite->order = malloc(balanced_count * sizeof(size_t));
    if (!dataset_meteorite->order)
    {
        free(fell_indices);
        free(found_indices);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    for (size_t i = 0; i < min_count; i++)
    {
        dataset_meteorite->order[2 * i] = fell_indices[i];
        dataset_meteorite->order[2 * i + 1] = found_indices[i];
    }

    float *balanced_inputs = malloc(balanced_count * METEORITE_FEATURES * sizeof(float));
    uint8_t *balanced_labels = malloc(balanced_count * sizeof(uint8_t));
    if (!balanced_inputs || !balanced_labels)
    {
        free(balanced_inputs);
        free(balanced_labels);
        free(fell_indices);
        free(found_indices);
        free(dataset_meteorite->order);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    for (size_t i = 0; i < balanced_count; i++)
    {
        size_t idx = dataset_meteorite->order[i];
        size_t src_base = idx * METEORITE_FEATURES;
        size_t dst_base = i * METEORITE_FEATURES;
        for (size_t j = 0; j < METEORITE_FEATURES; j++)
        {
            balanced_inputs[dst_base + j] = dataset_meteorite->inputs[src_base + j];
        }
        balanced_labels[i] = dataset_meteorite->labels[idx];
    }

    free(dataset_meteorite->inputs);
    free(dataset_meteorite->labels);
    free(dataset_meteorite->order);

    dataset_meteorite->inputs = balanced_inputs;
    dataset_meteorite->labels = balanced_labels;
    dataset_meteorite->count = balanced_count;

    dataset_meteorite->order = malloc(dataset_meteorite->count * sizeof(size_t));
    if (!dataset_meteorite->order)
    {
        free(fell_indices);
        free(found_indices);
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite);
        return NULL;
    }

    for (size_t i = 0; i < dataset_meteorite->count; i++)
    {
        dataset_meteorite->order[i] = i;
    }
    free(fell_indices);
    free(found_indices);

    static const DatasetOps ops = {
        .size = dataset_meteorite_size,
        .shuffle = dataset_meteorite_shuffle,
        .next_batch = dataset_meteorite_next_batch,
        .reset = dataset_meteorite_reset,
        .free = dataset_meteorite_free};

    Dataset *dataset = dataset_new(dataset_meteorite, &ops);
    if (!dataset)
    {
        free(dataset_meteorite->inputs);
        free(dataset_meteorite->labels);
        free(dataset_meteorite->order);
        free(dataset_meteorite);
        return NULL;
    }

    return dataset;
}
