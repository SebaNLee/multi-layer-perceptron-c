#include "mlp/loss/cross_entropy.h"

/**
 * @brief Cross Entropy loss
 *
 * Forward input:
 *  y_prediction
 *  y_label
 *
 * Forward computes:
 *  L = - sum_i y_label_i * log(y_prediction_i)
 *
 * Backward input:
 *  y_prediction
 *  y_label
 *
 * Backward computes:
 *  dL_i = y_prediction_i - y_label_i
 *
 * Shapes:
 *  y_prediction: (n, 1)
 *  y_label: (n, 1)
 */
typedef struct
{
    char _; // unused
} CrossEntropy;
