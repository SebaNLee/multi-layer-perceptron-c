#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct
{
    float *data;     // buffer
    size_t *shape;   // sizes per dimension
    size_t *strides; // steps in memory
    size_t rank;     // number of dimensions
    size_t size;     // total number of elements
} Tensor;

Tensor *tensor_new(size_t rank, size_t *shape);
void tensor_free(Tensor *tensor);
float tensor_get(Tensor *tensor, size_t *idx);
void tensor_set(Tensor *tensor, size_t *idx, float value);
static size_t tensor_offset(Tensor *tensor, size_t *idx);
void tensor_fill(Tensor *tensor, float value);
void tensor_zero(Tensor *tensor);
static bool tensor_same_shape(Tensor *a, Tensor *b);
void tensor_copy(Tensor *destination, Tensor *source);
float tensor_dot(Tensor *a, Tensor *b); // implemented for: rank 1; rank 2 shape n,1;
Tensor *tensor_transpose_2d(Tensor *tensor); // implementation with copy
