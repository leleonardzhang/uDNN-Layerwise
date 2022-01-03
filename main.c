/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"

static layer_t layer0 = {.class=Conv2d, .kernel=&LAYER_0_KERNEL_MAT, .bias=&LAYER_0_BIAS_MAT, .numChannels=1, .numFilters=8, .stride_numRows=1, .stride_numCols=1, .activation=&fp_linear, .padding=Same};
static layer_t layer1 = {.class=Maxpooling, .numChannels=8, .pool_numRows=3, .pool_numCols=3};
static layer_t layer2 = {.class=Conv2d, .kernel=&LAYER_2_KERNEL_MAT, .bias=&LAYER_2_BIAS_MAT, .numChannels=8, .numFilters=16, .stride_numRows=1, .stride_numCols=1, .activation=&fp_linear, .padding=Same};
static layer_t layer3 = {.class=Maxpooling, .numChannels=16, .pool_numRows=3, .pool_numCols=3};
static layer_t layer4 = {.class=Flatten, .numChannels=16};
static layer_t layer5 = {.class=Dense, .kernel=&LAYER_5_KERNEL_MAT, .bias=&LAYER_5_BIAS_MAT, .activation=&fp_linear};
static layer_t layer6 = {.class=Dense, .kernel=&LAYER_6_KERNEL_MAT, .bias=&LAYER_6_BIAS_MAT, .activation=&fp_linear};

static layer_t *layers[7] = {&layer0, &layer1, &layer2, &layer3, &layer4, &layer5, &layer6};

static model_t model = {.layers = layers, .numLayers=7, .input=&INPUT_BUFFER_MAT, .output=&OUTPUT_BUFFER_MAT};

void main(void){

    /* stop watchdog timer */
    WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

    predict(&model);
    LABEL = argmax(&OUTPUT_BUFFER_MAT);
    __no_operation();
}
