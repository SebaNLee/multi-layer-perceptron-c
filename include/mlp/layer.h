
typedef struct
{
    // TODO
    void (*feedforward)(Layer * self);
    void (*backpropagation)(Layer * self);
    void (*free)(Layer * self);
} LayerOps;

typedef struct
{
    LayerOps * ops;
    void * impl;
} Layer;