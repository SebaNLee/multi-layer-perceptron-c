#ifndef DATASET_DATASET_H
#define DATASET_DATASET_H

#include "mlp/tensor.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct Dataset Dataset;

typedef struct
{
    size_t (*size)(Dataset *self);
    void (*shuffle)(Dataset *self);
    bool (*next_batch)(Dataset *self, Tensor **inputs, Tensor **labels, size_t batch_size);
    void (*reset)(Dataset *self);
    void (*free)(Dataset *self);
} DatasetOps;

struct Dataset
{
    const DatasetOps *ops;
    void *impl;
};

Dataset *dataset_new(void *impl, const DatasetOps *ops);
void dataset_free(Dataset *dataset);

size_t dataset_size(Dataset *dataset);
void dataset_shuffle(Dataset *dataset);
bool dataset_next_batch(Dataset *dataset, Tensor **inputs, Tensor **labels, size_t batch_size);
void dataset_reset(Dataset *dataset);

#endif