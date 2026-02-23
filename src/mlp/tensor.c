#include "tensor.h"

/**
 * @name Internal helpers
 * @{
 */

static bool tensor_same_shape(const Tensor *tensor1, const Tensor *tensor2);
static size_t tensor_offset(const Tensor *tensor, const size_t *idx);

/** @} */

static bool tensor_same_shape(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2 || tensor1->rank != tensor2->rank)
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

static size_t tensor_offset(const Tensor *tensor, const size_t *idx)
{
    if (!tensor)
    {
        // TODO error
        return 0;
    }

    size_t offset = 0;
    for (size_t i = 0; i < tensor->rank; i++)
    {
        if (idx[i] < tensor->shape[i])
        {
            // TODO error
            return 0;
        }

        offset += idx[i] * tensor->strides[i];
    }

    return offset;
}

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

void tensor_copy(Tensor *destination, const Tensor *source)
{
    if (!destination || !source || !tensor_same_shape(destination, source))
    {
        return;
    }

    memcpy(destination->data, source->data, source->size * sizeof(float));
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

Tensor *tensor_add(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2 || !tensor_same_shape(tensor1, tensor2))
    {
        return NULL;
    }

    Tensor *result = tensor_new(tensor1->rank, tensor1->shape);
    if (!result)
    {
        return NULL;
    }

    for (size_t i = 0; i < tensor1->size; i++)
    {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }

    return result;
}

Tensor *tensor_sub(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2 || !tensor_same_shape(tensor1, tensor2))
    {
        return NULL;
    }

    Tensor *result = tensor_new(tensor1->rank, tensor1->shape);
    if (!result)
    {
        return NULL;
    }

    for (size_t i = 0; i < tensor1->size; i++)
    {
        result->data[i] = tensor1->data[i] - tensor2->data[i];
    }

    return result;
}

Tensor *tensor_mul(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2 || !tensor_same_shape(tensor1, tensor2))
    {
        return NULL;
    }

    Tensor *result = tensor_new(tensor1->rank, tensor1->shape);
    if (!result)
    {
        return NULL;
    }

    for (size_t i = 0; i < tensor1->size; i++)
    {
        result->data[i] = tensor1->data[i] * tensor2->data[i];
    }

    return result;
}

Tensor *tensor_scale(const Tensor *tensor, float scalar)
{
    if (!tensor)
    {
        return NULL;
    }

    Tensor *result = tensor_new(tensor->rank, tensor->shape);
    if (!result)
    {
        return NULL;
    }

    for (size_t i = 0; i < tensor->size; i++)
    {
        result->data[i] = tensor->data[i] * scalar;
    }

    return result;
}

float tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2)
    {
        return 0;
    }

    if (tensor1->rank != 1 || tensor2->rank != 1 || tensor1->shape[0] != tensor2->shape[0])
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

Tensor *tensor_outer_product(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2)
    {
        return NULL;
    }

    if (tensor1->rank != 1 || tensor2->rank != 1)
    {
        return NULL;
    }

    size_t result_shape[2] = {tensor1->shape[0], tensor2->shape[0]};
    Tensor *result = tensor_new(2, result_shape);
    if (!result)
    {
        return NULL;
    }

    size_t result_idx[2];

    for (size_t i = 0; i < tensor1->shape[0]; i++)
    {
        for (size_t j = 0; j < tensor2->shape[0]; j++)
        {
            result_idx[0] = i;
            result_idx[1] = j;
            tensor_set(result, result_idx, tensor1->data[i] * tensor2->data[j]);
        }
    }

    return result;
}

Tensor *tensor_matrix_multiplication(const Tensor *tensor1, const Tensor *tensor2)
{
    if (!tensor1 || !tensor2)
    {
        return NULL;
    }

    if (tensor1->rank != 2 || tensor2->rank != 2 || tensor1->shape[1] != tensor2->shape[0])
    {
        return NULL;
    }

    size_t result_shape[2] = {tensor1->shape[0], tensor2->shape[1]};
    Tensor *result = tensor_new(2, result_shape);
    if (!result)
    {
        return NULL;
    }

    for (size_t i = 0; i < tensor1->shape[0]; i++)
    {
        for (size_t j = 0; j < tensor2->shape[1]; j++)
        {
            float sum = 0;

            for (size_t k = 0; k < tensor1->shape[1]; k++)
            {
                size_t tensor1_idx[2] = {i, k};
                size_t tensor2_idx[2] = {k, j};

                sum += tensor_get(tensor1, tensor1_idx) * tensor_get(tensor2, tensor2_idx);
            }

            size_t result_idx[2] = {i, j};
            tensor_set(result, result_idx, sum);
        }
    }

    return result;
}

Tensor *tensor_transpose_2d(const Tensor *tensor)
{
    if (!tensor || tensor->rank != 2)
    {
        return NULL;
    }

    size_t result_shape[2] = {tensor->shape[1], tensor->shape[0]};

    Tensor *result = tensor_new(2, result_shape);
    if (!result)
    {
        return NULL;
    }

    size_t tensor_idx[2];
    size_t result_idx[2];

    for (size_t i = 0; i < tensor->shape[0]; i++)
    {
        for (size_t j = 0; j < tensor->shape[1]; j++)
        {
            tensor_idx[0] = i;
            tensor_idx[1] = j;
            result_idx[0] = j;
            result_idx[1] = i;

            tensor_set(result, result_idx, tensor_get(tensor, tensor_idx));
        }
    }

    return result;
}

Tensor *tensor_reshape(const Tensor *tensor, size_t rank, const size_t *shape)
{
    if (!tensor || rank == 0 || !shape)
    {
        return NULL;
    }

    size_t new_size = 1;
    for (size_t i = 0; i < rank; i++)
    {
        new_size *= shape[i];
    }

    if (new_size != tensor->size)
    {
        return NULL;
    }

    Tensor *result = tensor_new(rank, shape);
    if (!result)
    {
        return NULL;
    }

    memcpy(result->data, tensor->data, new_size * sizeof(float));

    return result;
}