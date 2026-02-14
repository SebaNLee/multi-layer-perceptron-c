#include <stddef.h>
#include "layer.h"

typedef struct
{
    Layer ** layers;
    size_t layers_count;
} MLP;