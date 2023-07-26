#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "neural_network.h"

enum {
	TYPE_XOR,
	TYPE_NXOR,
	TYPE_AND,
	TYPE_NAND,
	TYPE_OR,
	TYPE_NOR,
};

void print_help(const char *argv0);
void train_xor(NeuralNetwork *nn, int n, float rate);
void train_nxor(NeuralNetwork *nn, int n, float rate);
void train_and(NeuralNetwork *nn, int n, float rate);
void train_nand(NeuralNetwork *nn, int n, float rate);
void train_or(NeuralNetwork *nn, int n, float rate);
void train_nor(NeuralNetwork *nn, int n, float rate);

void show_output(NeuralNetwork *nn);

float input[4][2] =	{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
};

void
print_help(const char *argv0)
{
	printf("%s\n"
			"    -h for help.\n"
			"    -T <type> to specify training type\n"
			"    -n <train times> to specify how many times to train the neural network\n"
			"    -r <learning rate> to specify the learning rate\n"
			"    -f <file_name> to specify the save file.\n"
			"    -L to load save a neural network.\n"
			,
			argv0);
}

void
train_xor(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{0},
		{1},
		{1},
		{0},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}
}

void
train_nxor(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{1},
		{0},
		{0},
		{1},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}
}

void
train_and(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{0},
		{0},
		{0},
		{1},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}
}

void
train_nand(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{1},
		{1},
		{1},
		{0},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}

}

void
train_or(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{0},
		{1},
		{1},
		{1},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}
}

void
train_nor(NeuralNetwork *nn, int n, float rate)
{
	int i;
	int j;

	float expect[4][1] = {
		{1},
		{0},
		{0},
		{0},
	};

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 4; j++)
		{
			nn_train(nn, input[j], expect[j], rate);
		}
	}
}

void
show_output(NeuralNetwork *nn)
{
	int i;
	float *output;

	for (i = 0; i < 4; i++)
	{
		output = nn_run(nn, input[i]);
		printf("input: %6.2f, %6.2f, output: %6.2f\n", input[i][0], input[i][1], output[0]);
	}
}

int main(int argc, char **argv)
{
	NeuralNetwork *nn = NULL;
	int c;
	const char *f_name = NULL;
	int train_or_load = 0;
	int type = 0;
	int n = 10000;
	float rate = 0.1;

	if (argc < 2)
	{
		print_help(argv[0]);
		return 0;
	}

	while ((c = getopt(argc, argv, "hTLn:r:f:")) != -1)
	{
		switch (c)
		{
			case 'T':
				if (strcasecmp(optarg, "xor") == 0)
				{
					type = TYPE_XOR;
				}
				else if (strcasecmp(optarg, "nxor") == 0)
				{
					type = TYPE_NXOR;
				}
				else if (strcasecmp(optarg, "and") == 0)
				{
					type = TYPE_AND;
				}
				else if (strcasecmp(optarg, "nand") == 0)
				{
					type = TYPE_NAND;
				}
				else if (strcasecmp(optarg, "or") == 0)
				{
					type = TYPE_OR;
				}
				else if (strcasecmp(optarg, "nor") == 0)
				{
					type = TYPE_NOR;
				}
				else
				{
					printf("Unknown type.\n");
					return 1;
				}
				break;
			case 'n':
				n = atoi(optarg);
				break;
			case 'r':
				rate = atof(optarg);
				break;
			case 'f':
				f_name = optarg;
				break;
			case 'L':
				train_or_load = 1;
				break;
			case 'h':
			default:
				print_help(argv[0]);
				return 1;
		}
	}

	if (train_or_load == 0)
	{
		nn = nn_create(2,
				1,
				1,
				2,
				1,
				ACT_FUNC_TYPE_SIGMOID,
				ACT_FUNC_TYPE_SIGMOID);
		printf("Training...\n");

		switch (type)
		{
			case TYPE_XOR:
				train_xor(nn, n, rate);
				break;
			case TYPE_NXOR:
				train_nxor(nn, n, rate);
				break;
			case TYPE_AND:
				train_and(nn, n, rate);
				break;
			case TYPE_NAND:
				train_nand(nn, n, rate);
				break;
			case TYPE_OR:
				train_or(nn, n, rate);
				break;
			case TYPE_NOR:
				train_nor(nn, n, rate);
				break;
			default:
				printf("Unknown type.\n");
				nn_free(nn);
				return 1;
		}
	}
	else
	{
		if (f_name == NULL)
		{
			printf("Need a neural network file.\n");
			return 1;
		}

		printf("Loading %s...\n", f_name);
		nn = nn_load(f_name);

		if (nn == NULL)
		{
			printf("Failed  to load neural network.\n");
			return 1;
		}
	}

	show_output(nn);

	if (train_or_load == 0)
	{
		if (f_name != NULL)
		{
			if (nn_save(nn, f_name))
			{
				printf("Failed to save neural network to %s\n", f_name);
			}
		}
	}

	nn_free(nn);

	return 0;
}
