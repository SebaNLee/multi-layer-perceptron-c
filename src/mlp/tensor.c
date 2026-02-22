#include "tensor.h"

Tensor *tensor_new(size_t rank, size_t *shape)
{
    if (rank == 0 || !shape)
    {
        return NULL;
    }

    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor)
    {
        return NULL;
    }

    tensor->shape = malloc(rank * sizeof(size_t));
    tensor->strides = malloc(rank * sizeof(size_t));
    if (!tensor->shape || !tensor->strides)
    {
        tensor_free(tensor);
        return NULL;
    }

    tensor->size = 1;
    for (size_t i = 0; i < rank; i++)
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->rank = rank;
    tensor->strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i++)
    {
        tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
    }

    tensor->data = calloc(tensor->size, sizeof(float));
    if (!tensor->data)
    {
        tensor_free(tensor);
        return NULL;
    }

    return tensor;
}

void tensor_free(Tensor *tensor)
{
    if (!tensor)
    {
        return;
    }

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

float tensor_get(Tensor *tensor, size_t *idx)
{
    // TODO defensive params

    return tensor->data[tensor_offset(tensor, idx)];
}

void tensor_set(Tensor *tensor, size_t *idx, float value)
{
    // TODO defensive params

    tensor->data[tensor_offset(tensor, idx)] = value;
}

static size_t tensor_offset(Tensor *tensor, size_t *idx)
{
    size_t offset = 0;
    for (size_t i = 0; i < tensor->rank; i++)
    {
        offset += idx[i] * tensor->strides[i];
    }
    
    return offset;
}