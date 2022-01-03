#include <stdint.h>
#include "../math/matrix_ops.h"
#include "../math/matrix.h"

#ifndef LAYERS_LAYER_H_
#define LAYERS_LAYER_H_

typedef enum {Dense, Conv2d, Flatten, Maxpooling} Layer;
typedef enum {Valid, Same} Padding;

typedef struct{
    Layer class;
    matrix* kernel;
    matrix* bias;
    uint16_t numChannels;
    uint16_t numFilters;
    uint16_t stride_numRows;
    uint16_t stride_numCols;
    uint16_t pool_numRows;
    uint16_t pool_numCols;
    int16_t (*activation)(int16_t, uint16_t);
    Padding padding;
    bool trainable;
} layer_t;


typedef struct{
    layer_t **layers;
    uint16_t numLayers;
    matrix *input;
    matrix *output;
} model_t;

#endif /* LAYERS_LAYER_H_ */
