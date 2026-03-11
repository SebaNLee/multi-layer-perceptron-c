#ifndef EXAMPLES_EMNIST_LETTERS_DATASET_EMNIST_LETTERS_H
#define EXAMPLES_EMNIST_LETTERS_DATASET_EMNIST_LETTERS_H

#include "dataset/dataset.h"

typedef enum
{
    TRAIN,
    TEST
} EmnistSubset;

Dataset *dataset_emnist_letters_new(EmnistSubset subset);

#endif
