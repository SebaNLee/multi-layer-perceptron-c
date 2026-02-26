#include "mlp.h"

MLP *mlp_new()
{
    MLP *mlp = malloc(sizeof(MLP));
    if (!mlp)
    {
        return NULL;
    }

    mlp->layers_count = 0;
    mlp->layers_size = BLOCK;

    mlp->layers = malloc(BLOCK * sizeof(Layer *));
    if (!mlp->layers)
    {
        free(mlp);
        return NULL;
    }

    return mlp;
}

void mlp_free(MLP *mlp)
{
    if (!mlp)
    {
        return;
    }

    for (size_t i = 0; i < mlp->layers_count; i++)
    {
        layer_free(mlp->layers[i]);
    }

    free(mlp->layers);
    free(mlp);
}

void mlp_add_layer(MLP *mlp, Layer *layer)
{
    if (!mlp || !layer)
    {
        return;
    }

    if (mlp->layers_count == mlp->layers_size)
    {
        size_t new_layers_size = mlp->layers_size + BLOCK;
        Layer **new_layers = realloc(mlp->layers, new_layers_size * sizeof(Layer *));
        if (!new_layers)
        {
            return;
        }

        mlp->layers = new_layers;
        mlp->layers_size = new_layers_size;
    }

    mlp->layers[mlp->layers_count++] = layer;
}

Tensor *mlp_forward(MLP *mlp, Tensor *input)
{
    Tensor *current = input;

    for (size_t i = 0; i < mlp->layers_count; i++)
    {
        layer_forward(mlp->layers[i], current);
        current = mlp->layers[i]->output;
    }

    return current;
}

void mlp_backward(MLP *mlp, Tensor *gradient_output)
{
    Tensor *current_gradient = gradient_output;

    size_t i = mlp->layers_count;
    while (i--)
    {
        layer_backward(mlp->layers[i], current_gradient);
        current_gradient = mlp->layers[i]->gradient_input;
    }
}
