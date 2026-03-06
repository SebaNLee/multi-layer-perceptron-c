#include "mlp/loss/cross_entropy.h"
#include <math.h>

#define EPSILON 1e-7f

/**
 * @brief Cross Entropy loss
 *
 * Forward input:
 *  y_prediction
 *  y_label
 *
 * Forward computes:
 *  L = - (1/batch_size) * sum_i y_label_i * log(y_prediction_i)
 *
 * Backward input:
 *  y_prediction
 *  y_label
 *
 * Backward computes:
 *  dL_i = (1/batch_size) * (y_prediction_i - y_label_i)
 *
 * Shapes:
 *  y_prediction: (n, batch)
 *  y_label: (n, batch)
 */
typedef struct
{
    char _; // unused
} CrossEntropy;

static float loss_cross_entropy_forward(Loss *self, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!y_prediction || !y_label)
    {
        return 0;
    }

    float loss = 0;

    for (size_t i = 0; i < y_prediction->size; i++)
    {

        float y_prediction_i = y_prediction->data[i];

        if (y_prediction_i < EPSILON)
        {
            y_prediction_i = EPSILON;
        }

        loss -= y_label->data[i] * logf(y_prediction_i);
    }

    size_t batch_size = y_prediction->shape[1];

    return loss / batch_size;
}

static Tensor *loss_cross_entropy_backward(Loss *self, const Tensor *y_prediction, const Tensor *y_label)
{
    if (!y_prediction || !y_label)
    {
        return NULL;
    }

    Tensor *gradient = tensor_new(y_prediction->rank, y_prediction->shape);
    if (!gradient)
    {
        return NULL;
    }

    float inv_batch_size = 1 / (float)y_prediction->shape[1];

    for (size_t i = 0; i < gradient->size; i++)
    {
        gradient->data[i] = (y_prediction->data[i] - y_label->data[i]) * inv_batch_size;
    }

    return gradient;
}

static void loss_cross_entropy_free(Loss *self)
{
    if (!self || !self->impl)
    {
        return;
    }

    free(self->impl);
}

Loss *loss_cross_entropy_new(void)
{
    CrossEntropy *cross_entropy = calloc(1, sizeof(CrossEntropy));
    if (!cross_entropy)
    {
        return NULL;
    }

    static const LossOps ops = {
        .forward = loss_cross_entropy_forward,
        .backward = loss_cross_entropy_backward,
        .free = loss_cross_entropy_free};

    Loss *loss = loss_new(cross_entropy, &ops);
    if (!loss)
    {
        free(cross_entropy);

        return NULL;
    }

    return loss;
}