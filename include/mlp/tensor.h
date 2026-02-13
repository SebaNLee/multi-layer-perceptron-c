#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    float *data;     // buffer
    size_t *shape;   // sizes per dimension
    size_t *strides; // steps in memory
    size_t rank;     // number of dimensions
    size_t size;     // total number of elements
} Tensor;

Tensor *tensor_new(size_t rank, size_t *shape);
void tensor_free(Tensor *t);
float tensor_get(Tensor *t, size_t *idx);
void tensor_set(Tensor *t, size_t *idx, float value);
