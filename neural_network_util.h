#ifndef __NEURAL_NETWORK_UTIL_H
#define __NEURAL_NETWORK_UTIL_H

#include "neural_network.h"

/* For debug uses */
const float *nn_util_get_w_matrix(NeuralNetwork *nn, int layer, int *n_row, int *n_col);

void nn_util_matrix_print(const float *matrix, int n_row, int n_col);

const float *nn_util_get_output(NeuralNetwork *nn, int layer, int *n);

const float *nn_util_get_bias(NeuralNetwork *nn, int layer, int *n);

const float *nn_util_get_delta(NeuralNetwork *nn, int layer, int *n);

void nn_util_vector_print(const float *vector, int n);

/* For utilitiy */
int nn_util_find_most_possible(const float *output, int n);

#endif /* __NEURAL_NETWORK_H */
