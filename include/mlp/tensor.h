#include <stdbool.h>
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

Tensor *tensor_new(size_t rank, const size_t *shape);
void tensor_free(Tensor *tensor);

float tensor_get(const Tensor *tensor, const size_t *idx);
void tensor_set(Tensor *tensor, const size_t *idx, float value);
Tensor *tensor_clone(const Tensor *tensor);

void tensor_fill(Tensor *tensor, float value);
void tensor_zero(Tensor *tensor);

Tensor *tensor_add(const Tensor *tensor1, const Tensor *tensor2); // implementation: returns copy
Tensor *tensor_sub(const Tensor *tensor1, const Tensor *tensor2); // implementation: returns copy
Tensor *tensor_mul(const Tensor *tensor1, const Tensor *tensor2); // implementation: returns copy
Tensor *tensor_scale(const Tensor *tensor, float scalar);         // implementation: returns copy

float tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2);             // rank 1
Tensor *tensor_outer_product(const Tensor *tensor1, const Tensor *tensor2);         // rank 1
Tensor *tensor_matrix_multiplication(const Tensor *tensor1, const Tensor *tensor2); // rank 2

Tensor *tensor_transpose_2d(const Tensor *tensor);                              // implementation: returns copy
Tensor *tensor_reshape(const Tensor *tensor, size_t rank, const size_t *shape); // implementation: returns copy
