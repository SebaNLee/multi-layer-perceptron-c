
typedef struct
{
    // TODO
    void (*forward)(Layer * self);
    void (*backward)(Layer * self);
    void (*free)(Layer * self);
} LayerOps;

typedef struct
{
    LayerOps * ops;
    void * impl;
} Layer;