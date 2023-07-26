#include "../neural_network.h"
#include "../neural_network_util.h"

int main(void)
{
	NeuralNetwork *nn;
	float input[2] = {1, 1};
	const float *output;
	const float *w_matrix;
	int n_row;
	int n_col;
	int n;
	nn = nn_create(2, 1, 1, 3, 0, ACT_FUNC_TYPE_LINEAR, ACT_FUNC_TYPE_LINEAR);

	output = nn_run(nn, input);

	printf("Input: ");
	nn_util_vector_print(input, 2);

	printf("w_matrix layer 0:\n");
	w_matrix = nn_util_get_w_matrix(nn, 0, &n_row, &n_col);
	nn_util_matrix_print(w_matrix, n_row, n_col);

	printf("layer 0 output:\n");
	output = nn_util_get_output(nn, 0, &n);
	nn_util_vector_print(output, n);

	printf("w_matrix layer 1:\n");
	w_matrix = nn_util_get_w_matrix(nn, 1, &n_row, &n_col);
	nn_util_matrix_print(w_matrix, n_row, n_col);

	printf("layer 1 output:\n");
	output = nn_util_get_output(nn, 1, &n);
	nn_util_vector_print(output, n);


	nn_free(nn);
	return 0;
}
