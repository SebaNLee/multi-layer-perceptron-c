#include "mlp/loss/mean_squared_error.h"

typedef struct
{
    char _; // unused
} MeanSquaredError;

static float loss_mean_squared_error_forward(Loss *self, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!y_prediction || !y_label)
    {
        return 0;
    }

    float loss = 0;

    for (size_t i = 0; i < y_prediction->size; i++)
    {
        float error = y_prediction->data[i] - y_label->data[i];
        loss += 0.5 * error * error;
    }

    return loss;
}

static Tensor *loss_mean_squared_error_backward(Loss *self, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!y_prediction || !y_label)
    {
        return 0;
    }

    Tensor *gradient = tensor_new(y_prediction->rank, y_prediction->shape);
    if (!gradient)
    {
        return;
    }

    for (size_t i = 0; i < gradient->size; i++)
    {
        gradient->data[i] = y_prediction->data[i] - y_label->data[i];
    }

    return gradient;
}

static void loss_mean_squared_error_free(Loss *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    free(self->impl);

    return;
}

Loss *loss_mean_squared_error_new(void)
{
    MeanSquaredError *mean_squared_error = calloc(1, sizeof(MeanSquaredError));
    if (!mean_squared_error)
    {
        return NULL;
    }

    static const LossOps ops = {
        .forward = loss_mean_squared_error_forward,
        .backward = loss_mean_squared_error_backward,
        .free = loss_mean_squared_error_free};

    Loss *loss = loss_new(mean_squared_error, &ops);
    if (!loss)
    {
        free(mean_squared_error);

        return NULL;
    }

    return loss;
}