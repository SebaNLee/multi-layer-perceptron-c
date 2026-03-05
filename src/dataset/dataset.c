#include "dataset/dataset.h"
#include <stdlib.h>

Dataset *dataset_new(void *impl, const DatasetOps *ops)
{
    if (!impl || !ops || !ops->size || !ops->next_batch)
    {
        return NULL;
    }

    Dataset *dataset = malloc(sizeof(Dataset));
    if (!dataset)
    {
        return NULL;
    }

    dataset->ops = ops;
    dataset->impl = impl;

    return dataset;
}

void dataset_free(Dataset *dataset)
{
    if (!dataset)
    {
        return;
    }

    if (dataset->ops && dataset->ops->free)
    {
        dataset->ops->free(dataset);
    }

    free(dataset);
}

size_t dataset_size(Dataset *dataset)
{
    if (!dataset || !dataset->ops || !dataset->ops->size)
    {
        return 0;
    }

    return dataset->ops->size(dataset);
}

void dataset_shuffle(Dataset *dataset)
{
    if (!dataset || !dataset->ops || !dataset->ops->shuffle)
    {
        return;
    }

    dataset->ops->shuffle(dataset);
}

bool dataset_next_batch(Dataset *dataset, Tensor **inputs, Tensor **labels, size_t batch_size)
{
    if (!dataset || !dataset->ops || !dataset->ops->next_batch || !inputs || !labels)
    {
        return false;
    }

    return dataset->ops->next_batch(dataset, inputs, labels, batch_size);
}

void dataset_reset(Dataset *dataset)
{
    if (!dataset || !dataset->ops || !dataset->ops->reset)
    {
        return;
    }

    dataset->ops->reset(dataset);
}
