#include "neural_network_util.h"

static int nn_compute_vector_pos(NeuralNetwork *nn, int layer, int *n)
{
	int i;
	int pos;

	pos = 0;
	/* So many output for the 1-st layer */
	*n = nn->n_neuro_per_hidden;

	/* if n_hidden == 0 then this loop will be skipped, leaving pos = 0 */
	for (i = 0; i < layer; i++)
	{
		pos += *n;
	}

	/* If i happens to stop at the n_hidden,
	 * then so many output for the output layer */
	if (i == nn->n_hidden)
		*n = nn->n_output;

	return pos;
}

const float *
nn_util_get_w_matrix(NeuralNetwork *nn, int layer, int *n_row, int *n_col)
{
	int i;
	int pos;

	if (layer > nn->n_hidden ||
		layer < 0)
	{
		*n_col = 0;
		*n_row = 0;
		return NULL;
	}

	pos = 0;
	/* So many output for the 1-st layer */
	*n_row = nn->n_neuro_per_hidden;
	/* So many input for the 1-st layer */
	*n_col = nn->n_input;

	/* if n_hidden == 0 then this loop will be skipped, leaving pos = 0 */
	for (i = 0; i < layer; i++)
	{
		/* Move to next layer according to previous layer's input/output number */
		pos += *n_col * *n_row;

		/* So many output for the next layer */
		*n_row = nn->n_neuro_per_hidden;

		/* So many input for the next layer */
		*n_col = nn->n_neuro_per_hidden;
	}

	/* If i happens to stop at the n_hidden,
	 * then so many output for the output layer */
	if (i == nn->n_hidden)
		*n_row = nn->n_output;

	return &nn->weight[pos];
}

void
nn_util_matrix_print(const float *matrix, int n_row, int n_col)
{
	int r;
	int c;
	for(r = 0; r < n_row; r++)
	{
		putchar(r == 0 || r == n_row - 1 ? '+' : '|');
		for (c = 0; c < n_col; c++)
		{
			if (c != 0)
				printf(", ");
			printf("%6.2f", matrix[r * n_col + c]);
		}
		putchar(r == 0 || r == n_row - 1 ? '+' : '|');
		putchar('\n');
	}
}

const float *
nn_util_get_output(NeuralNetwork *nn, int layer, int *n)
{
	int pos;

	if (layer > nn->n_hidden ||
		layer < 0)
		return NULL;

	/* Call the same function since delta/bias/output are arranged the same way */
	pos = nn_compute_vector_pos(nn, layer, n);
	
	return &nn->output[pos];
}

const float *
nn_util_get_bias(NeuralNetwork *nn, int layer, int *n)
{
	int pos;

	if (layer > nn->n_hidden ||
		layer < 0)
		return NULL;

	/* Call the same function since delta/bias/output are arranged the same way */
	pos = nn_compute_vector_pos(nn, layer, n);
	
	return &nn->bias[pos];
}

const float *
nn_util_get_delta(NeuralNetwork *nn, int layer, int *n)
{
	int i;
	int pos;

	if (layer > nn->n_hidden ||
		layer < 0)
		return NULL;

	/* Call the same function since delta/bias/output are arranged the same way */
	pos = nn_compute_vector_pos(nn, layer, n);
	
	return &nn->bias[pos];
}

void
nn_util_vector_print(const float *vector, int n)
{
	int i;
	putchar('[');
	for (i = 0; i < n; i++)
	{
		if (i != 0)
			printf(", ");
		printf("%6.2f", vector[i]);
	}
	putchar(']');
	putchar('\n');
}

/* For utilitiy */
int
nn_util_find_most_possible(const float *output, int n)
{
	int i;
	int i_max;
	float o_max;

	i_max = 0;
	o_max = output[0];
	for (i = 1; i < n; i++)
	{
		if (o_max < output[i])
		{
			o_max = output[i];
			i_max = i;
		}
	}
	
	return i_max;
}
