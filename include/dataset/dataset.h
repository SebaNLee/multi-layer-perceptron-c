#include <stddef.h>

typedef struct
{
    // TODO
    size_t (*size)(Dataset * self);
    void (*shuffle)(Dataset * self);
    void (*free)(Dataset * self);
} DatasetOps;

typedef struct
{
    DatasetOps * ops;
    void * impl;
} Dataset;
