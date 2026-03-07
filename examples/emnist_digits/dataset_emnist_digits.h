#ifndef EXAMPLES_EMNIST_DIGITS_DATASET_EMNIST_DIGITS_H
#define EXAMPLES_EMNIST_DIGITS_DATASET_EMNIST_DIGITS_H

#include "dataset/dataset.h"

typedef enum
{
    TRAIN,
    TEST
} EmnistSubset;

Dataset *dataset_emnist_digits_new(EmnistSubset subset);

#endif
