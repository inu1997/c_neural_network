#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float nn_gen_random();

static int nn_compute_n_weight(NeuralNetwork *nn);

static void nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *bias, float *weight);

static void nn_correct(float *weight, float *delta, float *input, int n_input, int n_output, float rate);

static float
nn_gen_random()
{
    float r;

    r = rand();             /* A random 0 ~ RAND_MAX */
    r /= (float)RAND_MAX;   /* A random 0 ~ 1.0 */
    r -= 0.5;               /* A random -0.5 ~ 0.5 */

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

static void
nn_forward_propagation(NeuralNetwork *nn, float *input, int n_input, float *output, int n_output, float *bias, float *weight)
{
    int i; 
    int j;

    for (i = 0; i < n_output; i++)
    {
        output[i] = bias[i];
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

    nn->bias = calloc(nn->_n_neuro, sizeof(float));
    nn->output = calloc(nn->_n_neuro, sizeof(float));
    nn->delta = calloc(nn->_n_neuro, sizeof(float));
    nn->weight = calloc(nn->_n_weight, sizeof(float));

    nn_randomize(nn);

    return nn;
}

void
nn_free(NeuralNetwork *nn)
{
    free(nn->bias);
    free(nn->output);
    free(nn->delta);
    free(nn->weight);
    free(nn);
}

NeuralNetwork *
nn_duplicate(NeuralNetwork *nn)
{
    NeuralNetwork *new_nn;
    new_nn = nn_create(nn->n_input, nn->n_output, nn->n_hidden, nn->n_neuro_per_hidden, nn->act_func_type);

    memcpy(new_nn->weight, nn->weight, nn->_n_weight);
    memcpy(new_nn->bias, nn->bias, nn->_n_neuro);

    return new_nn;
}

void
nn_copy(NeuralNetwork *src, NeuralNetwork *dest)
{
    if (dest->n_input != src->n_input)
        return;
    if (dest->n_output != src->n_output)
        return;
    if (dest->n_hidden != src->n_hidden)
        return;
    if (dest->n_neuro_per_hidden != src->n_neuro_per_hidden)
        return;
    if (dest->act_func_type != src->act_func_type)
        return;
    
    memcpy(dest->weight, src->weight, src->_n_weight);
    memcpy(dest->bias, src->bias, src->_n_neuro);
}

float *
nn_run(NeuralNetwork *nn, float *input)
{
    int i;
    int j;
    int k;
    float *output;  /* Output buffer of this layer */
    float *bias;    /* Bias of this layer */
    float *weight;  /* Weight matrix of this layer */
    int n_input;    /* Number of input or Number of output of previous layer */
    int n_output;   /* Number of output of this layer */

    n_input = nn->n_input;
    output = nn->output;
    bias = nn->bias;
    weight = nn->weight;
    /*
     * 1. Process the hidden layers if any
     */
    for (i = 0; i < nn->n_hidden; i++)
    {
        /* So many outputs this layer */
        n_output = nn->n_neuro_per_hidden;
        /* Forward propergation */
        nn_forward_propagation(nn, input, n_input, output, n_output, bias, weight);

        /* Move pointer forward to the next layer */
        input = output; /* Output of this layer is the next layer's input */
        output += n_output;             /* Forwrad to the next layer */
        bias += n_output;
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
    nn_forward_propagation(nn, input, n_input, output, n_output, bias, weight);
    
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
    float *bias;        /* Bias of this layer */
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
    bias = &nn->bias[nn->_n_neuro - nn->n_output];
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
            
        bias[i] += delta[i] * rate;
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
        bias -= n_output;
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

            bias[j] += delta[j] * rate;
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
    bias -= n_output;
    output = input; /* Input is treated as the output of this "input layer" */
    
    /*
     * Correct the next layer's weight
     */
    nn_correct(next_weight, next_delta, output, n_output, n_next_output, rate);
    return ret;
}

void
nn_randomize(NeuralNetwork *nn)
{
    int i;
    for (i = 0; i < nn->_n_neuro; i++)
    {
        nn->bias[i] += nn_gen_random();
    }
    for (i = 0; i < nn->_n_weight; i++)
    {
        nn->weight[i] += nn_gen_random();
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
    if (fwrite(nn->bias, sizeof(float), nn->_n_neuro, f) < 0)
        goto __exit;
    if (fwrite(nn->weight, sizeof(float), nn->_n_weight, f) < 0)
        goto __exit;

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

    nn->bias = malloc(nn->_n_neuro * sizeof(float));
    nn->output = malloc(nn->_n_neuro * sizeof(float));
    nn->delta = malloc(nn->_n_neuro * sizeof(float));
    nn->weight = malloc(nn->_n_weight * sizeof(float));

    /* read weight and bias */
    if (fread(nn->bias, sizeof(float), nn->_n_neuro, f) < 0)
        goto __exit;
    if (fread(nn->weight, sizeof(float), nn->_n_weight, f) < 0)
        goto __exit;

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