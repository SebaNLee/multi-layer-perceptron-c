#include "tensor.h"

/**
 * @brief ReLU activation layer
 *
 * Forward input:
 *  X
 *
 * Forward computes:
 *  A = max(0, X)
 *
 * Backward input:
 *  dA = gradient_output
 *
 * Backward computes:
 *  dX = dA if X > 0
 *  dX = 0  if 0
 *
 * Shapes:
 *  X: (n, 1)
 *  A: (n, 1)
 */

typedef struct
{
    char _; // unused
} ReLU;
