#include "tensor.h"

Tensor *tensor_new(size_t rank, size_t *shape)
{
    if (rank == 0 || !shape)
    {
        return NULL;
    }

    Tensor *t = malloc(sizeof(Tensor));
    if (!t)
    {
        return NULL;
    }

    t->shape = malloc(rank * sizeof(size_t));
    t->strides = malloc(rank * sizeof(size_t));
    if (!t->shape || !t->strides)
    {
        tensor_free(t);
        return NULL;
    }

    t->size = 1;
    for (size_t i = 0; i < rank; i++)
    {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    t->rank = rank;
    t->strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i++)
    {
        t->strides[i - 1] = t->strides[i] * t->shape[i];
    }

    t->data = calloc(t->size, sizeof(float));
    if (!t->data)
    {
        tensor_free(t);
        return NULL;
    }

    return t;
}

void tensor_free(Tensor *t)
{
    if (!t)
    {
        return;
    }

    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

float tensor_get(Tensor *t, size_t *idx)
{
    // TODO defensive params

    return t->data[tensor_offset(t, idx)];
}

void tensor_set(Tensor *t, size_t *idx, float value)
{
    // TODO defensive params

    t->data[tensor_offset(t, idx)] = value;
}

static size_t tensor_offset(Tensor *t, size_t *idx)
{
    size_t offset = 0;
    for (size_t i = 0; i < t->rank; i++)
    {
        offset += idx[i] * t->strides[i];
    }
    
    return offset;
}