#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int random_pick(float rate);

static float nn_gen_random();

static float nn_gen_random_zero_to_one();

static int nn_compute_n_weight(NeuralNetwork *nn);

#if USE_BIAS
static void nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *bias, float *weight);
#else
static void nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *weight);
#endif

static void nn_correct(float *weight, float *delta, float *input, int n_input, int n_output, float rate);

static int
random_pick(float rate)
{
	if (nn_gen_random_zero_to_one() < rate)
		return 1;
	return 0;
}

static float
nn_gen_random()
{
    return nn_gen_random_zero_to_one() - 0.5f;	/* A random -0.5 ~ 0.5 */
}

static float nn_gen_random_zero_to_one()
{
    float r;

    r = rand();             /* A random 0 ~ RAND_MAX */
    r /= (float)RAND_MAX;   /* A random 0 ~ 1.0 */

	return r;
}

static int
nn_compute_n_weight(NeuralNetwork *nn)
{
    int n_weight;
    int i;
    int n_input;

    n_input = nn->n_input;
    n_weight = 0;
    for (i = 0; i < nn->n_hidden; i++)
    {
        n_weight += n_input * nn->n_neuro_per_hidden;
        n_input = nn->n_neuro_per_hidden;
    }
    
    n_weight += n_input * nn->n_output;

    return n_weight;
}

#if USE_BIAS
static void
nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *bias, float *weight)
#else
static void
nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *weight)
#endif
{
    int i; 
    int j;

    for (i = 0; i < n_output; i++)
    {
#if USE_BIAS
        output[i] = bias[i];
#else
        output[i] = 0;
#endif
        /* w vector dot i vector + bias */
        for (j = 0; j < n_input; j++)
        {
            output[i] += weight[i * n_input + j] * input[j];
        }
        /* Do activation function */
        if (nn->act_func_type == ACT_FUNC_TYPE_SIGMOID)
            output[i] = 1.0f / (1.0f + exp(-output[i]));
    }
}

static void
nn_correct(float *weight, float *delta, float *input, int n_input, int n_output, float rate)
{
    int i;
    int j;
    for (i = 0; i < n_output; i++)
    {
        for (j = 0; j < n_input; j++)
        {
            weight[i * n_input + j] += delta[i] * input[j] * rate;
        }
    }

}

NeuralNetwork *
nn_create(int n_input, int n_output, int n_hidden, int n_neuro_per_hidden, ACT_FUNC_TYPE act_func_type)
{
    int i;
    NeuralNetwork *nn;

    /* Error check */
    if (n_input < 0)
        return NULL;
    if (n_output < 0)
        return NULL;
    if (n_hidden < 0)
        return NULL;
    if (n_hidden > 0 && n_neuro_per_hidden < 1)
        return NULL;

    nn = malloc(sizeof(*nn));
    nn->n_input = n_input;
    nn->n_output = n_output;
    nn->n_hidden = n_hidden;
    nn->n_neuro_per_hidden = n_neuro_per_hidden;
    nn->act_func_type = act_func_type;
    /* Calculate number of neuro */
    nn->_n_neuro = n_output + n_hidden  * n_neuro_per_hidden;
    nn->_n_weight = nn_compute_n_weight(nn);

    nn->weight = malloc(nn->_n_weight * sizeof(float));
#if USE_BIAS
    nn->bias = malloc(nn->_n_neuro * sizeof(float));
#endif
    nn->output = malloc(nn->_n_neuro * sizeof(float));
    nn->delta = malloc(nn->_n_neuro * sizeof(float));

    nn_randomize(nn);

    return nn;
}

NeuralNetwork *
nn_produce(NeuralNetwork *a, NeuralNetwork *b)
{
	int i;
	NeuralNetwork *nn;

	if (a->n_input != b->n_input)
		return NULL;
	if (a->n_output != b->n_output)
		return NULL;
	if (a->n_hidden != b->n_hidden)
		return NULL;
	if (a->n_neuro_per_hidden != b->n_neuro_per_hidden)
		return NULL;
	if (a->act_func_type != b->act_func_type)
		return NULL;

	nn = nn_create(a->n_input, a->n_output, a->n_hidden, a->n_neuro_per_hidden, a->act_func_type);

#if USE_BIAS
	for (i = 0; i < a->_n_neuro; i++)
	{
		nn->bias[i] = rand() & 1 ? a->bias[i] : b->bias[i];
	}
#endif

	for (i = 0; i < a->_n_weight; i++)
	{
		nn->weight[i] = rand() & 1 ? a->weight[i] : b->weight[i];
	}

	return nn;
}

void
nn_free(NeuralNetwork *nn)
{
    free(nn->weight);
#if USE_BIAS
    free(nn->bias);
#endif
    free(nn->output);
    free(nn->delta);
    free(nn);
}

NeuralNetwork *
nn_duplicate(NeuralNetwork *nn)
{
    NeuralNetwork *new_nn;
    new_nn = nn_create(nn->n_input, nn->n_output, nn->n_hidden, nn->n_neuro_per_hidden, nn->act_func_type);

    memcpy(new_nn->weight, nn->weight, nn->_n_weight * sizeof(float));
#if USE_BIAS
    memcpy(new_nn->bias, nn->bias, nn->_n_neuro * sizeof(float));
#endif

    return new_nn;
}

float *
nn_run(NeuralNetwork *nn, float *input)
{
    int i;
    int j;
    int k;
    float *output;  /* Output buffer of this layer */
#if USE_BIAS
    float *bias;    /* Bias of this layer */
#endif
    float *weight;  /* Weight matrix of this layer */
    int n_input;    /* Number of input or Number of output of previous layer */
    int n_output;   /* Number of output of this layer */

    n_input = nn->n_input;
    output = nn->output;
#if USE_BIAS
    bias = nn->bias;
#endif
    weight = nn->weight;
    /*
     * 1. Process the hidden layers if any
     */
    for (i = 0; i < nn->n_hidden; i++)
    {
        /* So many outputs this layer */
        n_output = nn->n_neuro_per_hidden;
        /* Forward propergation */
#if USE_BIAS
        nn_forward_propagation(nn, input, n_input, output, n_output, bias, weight);
#else
        nn_forward_propagation(nn, input, n_input, output, n_output, weight);
#endif

        /* Move pointer forward to the next layer */
        input = output; /* Output of this layer is the next layer's input */
        output += n_output;             /* Forwrad to the next layer */
#if USE_BIAS
        bias += n_output;
#endif
        weight += n_input * n_output;   /* Forward to the next layer */
        /* Set the number of input to the previous layer */
        n_input = nn->n_neuro_per_hidden;
    }

    /*
     * 2. Process the output layer.
     */
    /* So many outputs this layer */
    n_output = nn->n_output;
    /* Forward propergation */
#if USE_BIAS
    nn_forward_propagation(nn, input, n_input, output, n_output, bias, weight);
#else
    nn_forward_propagation(nn, input, n_input, output, n_output, weight);
#endif
    
    return output;
}

float *
nn_train(NeuralNetwork *nn, float *input, float *expect, float rate)
{
    int i;
    int j;
    int k;
    float *ret;
    int n_output;       /* Number of output of this layer */
    int n_next_output;   /* Number of the neuro of next layer */
    float *delta;       /* Delta of this layer */
    float *output;      /* Output of this layer */
#if USE_BIAS
    float *bias;        /* Bias of this layer */
#endif
    float *next_delta;  /* delta of next layer */
    float *next_weight; /* delta of next layer */

    /*
     * 0. Run once
     */
    ret = nn_run(nn, input);

    /*
     * 1. From the output layer, do back propagation computation.
     */
    n_output = nn->n_output;
    output = &nn->output[nn->_n_neuro - nn->n_output];
#if USE_BIAS
    bias = &nn->bias[nn->_n_neuro - nn->n_output];
#endif
    delta = &nn->delta[nn->_n_neuro - nn->n_output];
    
    /*
     * Compute delta of this layer, also fix bias of this layer
     */
    for (i = 0; i < n_output; i++)
    {
        delta[i] = expect[i] - output[i];

        /* Apply derivation of activation function of this neuro */
        if (nn->act_func_type == ACT_FUNC_TYPE_SIGMOID)
            delta[i] *= output[i] * (1 - output[i]);
            
#if USE_BIAS
        bias[i] += delta[i] * rate;
#endif
    }

    /*
     * 2. From the last hidden layer, do back propagation computation
     */
    next_weight = &nn->weight[nn->_n_weight];
    for (i = 0; i < nn->n_hidden; i++)
    {
        n_next_output = n_output;
        n_output = nn->n_neuro_per_hidden;
        /* Move weight to this layer */
        next_weight -= n_next_output * n_output;

        /* Move next_delta, delta, output to this layer */
        next_delta = delta;
        delta -= n_output;
#if USE_BIAS
        bias -= n_output;
#endif
        output -= n_output;

        /*
         * a. Compute delta of this layer, also fix bias of this layer
         */
        for (j = 0; j < n_output; j++)
        {
            /*
             * The j-th neuro's delta is
             * "the next layer's delta" dot t"he j-th column vector of the next layer's weight matrix"
             * times the derivation of this neuro
             */
            delta[j] = 0;
            for (k = 0; k < n_next_output; k++)
            {
                delta[j] += next_delta[k] * next_weight[k * n_output + j];
            }

            /* Apply derivation of this neuro */
            if (nn->act_func_type == ACT_FUNC_TYPE_SIGMOID)
                delta[j] *= output[j] * (1 - output[j]);

#if USE_BIAS
            bias[j] += delta[j] * rate;
#endif
        }

        /*
         * b. Correct the next layer's weight
         */
        nn_correct(next_weight, next_delta, output, n_output, n_next_output, rate);
    }

    n_next_output = n_output;
    n_output = nn->n_input;
    /* Move weight to this layer */
    next_weight -= n_next_output * n_output;

    /* Move next_delta, output to this layer */
    next_delta = delta;
    output = input; /* Input is treated as the output of this "input layer" */
    
    /*
     * Correct the next layer's weight
     */
    nn_correct(next_weight, next_delta, output, n_output, n_next_output, rate);
    return ret;
}

void
nn_plus_randomize(NeuralNetwork *nn, float range)
{
    int i;

#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
        nn->bias[i] += nn_gen_random() * 2 * range;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
        nn->weight[i] += nn_gen_random() * 2 * range;
    }
}

void
nn_plus_randomize_by_rate(NeuralNetwork *nn, float range, float rate)
{
    int i;

#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
		if (random_pick(rate))
			nn->bias[i] += nn_gen_random() * 2 * range;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
		if (random_pick(rate))
			nn->weight[i] += nn_gen_random() * 2 * range;
    }

}

void
nn_randomize(NeuralNetwork *nn)
{
    int i;

#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
        nn->bias[i] = nn_gen_random() * 2;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
        nn->weight[i] = nn_gen_random() * 2;
    }
}

void
nn_randomize_with_scale(NeuralNetwork *nn, float scale)
{
    int i;

#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
        nn->bias[i] = nn_gen_random() * 2 * scale;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
        nn->weight[i] = nn_gen_random() * 2 * scale ;
    }
}

void
nn_randomize_by_rate(NeuralNetwork *nn, float rate)
{
    int i;

#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
		if (random_pick(rate))
			nn->bias[i] = nn_gen_random() * 2;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
		if (random_pick(rate))
			nn->weight[i] = nn_gen_random() * 2;
    }
}

void
nn_randomize_with_scale_by_rate(NeuralNetwork *nn, float scale, float rate)
{
    int i;
#if USE_BIAS
    for (i = 0; i < nn->_n_neuro; i++)
    {
		if (random_pick(rate))
			nn->bias[i] = nn_gen_random() * 2 * scale;
    }
#endif

    for (i = 0; i < nn->_n_weight; i++)
    {
		if (random_pick(rate))
			nn->weight[i] = nn_gen_random() * 2 * scale;
    }

}

int
nn_save(NeuralNetwork *nn, const char * file_name)
{
    FILE *f;
    int ret = -1;

    f = fopen(file_name, "wb+");
    if (f == NULL)
        return -1;
    
    /* write first informations */
    if (fwrite(&nn->n_input, sizeof(nn->n_input), 1, f) < 0)
        goto __exit;
    if (fwrite(&nn->n_output, sizeof(nn->n_output), 1, f) < 0)
        goto __exit;
    if (fwrite(&nn->n_hidden, sizeof(nn->n_hidden), 1, f) < 0)
        goto __exit;
    if (fwrite(&nn->n_neuro_per_hidden, sizeof(nn->n_neuro_per_hidden), 1, f) < 0)
        goto __exit;
    if (fwrite(&nn->act_func_type, sizeof(nn->act_func_type), 1, f) < 0)
        goto __exit;

    /* write weight and bias */
    if (fwrite(nn->weight, sizeof(float), nn->_n_weight, f) < 0)
        goto __exit;
#if USE_BIAS
    if (fwrite(nn->bias, sizeof(float), nn->_n_neuro, f) < 0)
        goto __exit;
#endif

    ret = 0;
__exit:
    fclose(f);
    return ret;
}

NeuralNetwork *
nn_load(const char *file_name)
{
    NeuralNetwork *nn;
    FILE *f;
    int is_ok = 0;

    f = fopen(file_name, "rb+");
    if (f == NULL)
        return NULL;
    
    nn = malloc(sizeof(*nn));

    /* read first informations */
    if (fread(&nn->n_input, sizeof(nn->n_input), 1, f) < 0)
        goto __exit;
    if (fread(&nn->n_output, sizeof(nn->n_output), 1, f) < 0)
        goto __exit;
    if (fread(&nn->n_hidden, sizeof(nn->n_hidden), 1, f) < 0)
        goto __exit;
    if (fread(&nn->n_neuro_per_hidden, sizeof(nn->n_neuro_per_hidden), 1, f) < 0)
        goto __exit;
    if (fread(&nn->act_func_type, sizeof(nn->act_func_type), 1, f) < 0)
        goto __exit;

    nn->_n_neuro = nn->n_output + nn->n_hidden  * nn->n_neuro_per_hidden;
    nn->_n_weight = nn_compute_n_weight(nn);

#if USE_BIAS
    nn->bias = malloc(nn->_n_neuro * sizeof(float));
#endif
    nn->output = malloc(nn->_n_neuro * sizeof(float));
    nn->delta = malloc(nn->_n_neuro * sizeof(float));
    nn->weight = malloc(nn->_n_weight * sizeof(float));

    /* read weight and bias */
    if (fread(nn->weight, sizeof(float), nn->_n_weight, f) < 0)
        goto __exit;
#if USE_BIAS
    if (fread(nn->bias, sizeof(float), nn->_n_neuro, f) < 0)
        goto __exit;
#endif

    is_ok = 1;
__exit:
    if (!is_ok)
    {
        nn_free(nn);
        nn = NULL;
    }
    fclose(f);
    return nn;
}
