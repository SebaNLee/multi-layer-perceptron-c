#include "tensor.h"

/**
 * @name Internal helpers
 * @{
 */

static bool tensor_same_shape(const Tensor *tensor1, const Tensor *tensor2);
static size_t tensor_offset(const Tensor *tensor, const size_t *idx);

/** @} */

Tensor *tensor_new(size_t rank, const size_t *shape)
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

float tensor_get(const Tensor *tensor, const size_t *idx)
{
    // TODO defensive params

    return tensor->data[tensor_offset(tensor, idx)];
}

void tensor_set(Tensor *tensor, const size_t *idx, float value)
{
    // TODO defensive params

    tensor->data[tensor_offset(tensor, idx)] = value;
}

static size_t tensor_offset(const Tensor *tensor, const size_t *idx)
{
    size_t offset = 0;
    for (size_t i = 0; i < tensor->rank; i++)
    {
        offset += idx[i] * tensor->strides[i];
    }

    return offset;
}

void tensor_fill(Tensor *tensor, float value)
{
    for (size_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
}

void tensor_zero(Tensor *tensor)
{
    memset(tensor->data, 0, tensor->size * sizeof(float));
}

static bool tensor_same_shape(const Tensor *tensor1, const Tensor *tensor2)
{
    if (tensor1->rank != tensor2->rank)
    {
        return false;
    }

    for (size_t i = 0; i < tensor1->rank; i++)
    {
        if (tensor1->shape[i] != tensor2->shape[i])
        {
            return false;
        }
    }

    return true;
}

void tensor_copy(Tensor *destination, const Tensor *source)
{
    if (!tensor_same_shape(destination, source))
    {
        return;
    }

    memcpy(destination->data, source->data, source->size * sizeof(float));
}

float tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2)
    {
        return 0;
    }

    if (tensor1->rank != 1 || tensor2->rank != 1)
    {
        return 0;
    }

    float sum = 0;

    for (size_t i = 0; i < tensor1->shape[0]; i++)
    {
        sum += tensor1->data[i] * tensor2->data[i];
    }

    return sum;
}

Tensor *tensor_transpose_2d

    (const Tensor *tensor)
{
    if (!tensor || tensor->rank != 2)
    {
        return NULL;
    }

    size_t transpose_shape[2] = {tensor->shape[1], tensor->shape[0]};

    Tensor *transpose_tensor = tensor_new(2, transpose_shape);
    if (!transpose_tensor)
    {
        return NULL;
    }

    size_t tensor_idx[2];
    size_t transpose_idx[2];

    for (size_t i = 0; i < tensor->shape[0]; i++)
    {
        for (size_t j = 0; j < tensor->shape[1]; j++)
        {
            tensor_idx[0] = i;
            tensor_idx[1] = j;
            transpose_idx[0] = j;
            transpose_idx[1] = i;

            tensor_set(transpose_tensor, transpose_idx, tensor_get(tensor, tensor_idx));
        }
    }

    return transpose_tensor;
}
