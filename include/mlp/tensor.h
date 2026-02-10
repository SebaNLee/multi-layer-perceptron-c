#include <stddef.h>

typedef struct
{
    float * data; // buffer
    size_t * shape; // sizes per dimension
    size_t rank; // number of dimensions
    size_t size; // total number of elements
} Tensor;
