#include "predict.h"

matrix *predict(model_t *model){
    uint16_t i = 0;
    matrix *input = &BUFFER_0_MAT, *output = &BUFFER_1_MAT;
    for (i = 0; i < model->numLayers; i ++){
        layer_t *layer = model->layers[i];
        if (i == 0) {
            memcpy(input->data, model->input->data, sizeof(dtype) * model->input->numCols * model->input->numRows);
            input->numRows = model->input->numRows;
            input->numCols = model->input->numCols;
        }
        switch (layer->class){
            case Dense:
                output->numRows = layer->kernel->numRows;
                output->numCols = input->numCols;
                dense(output, input, layer->kernel,  layer->bias, layer->activation, FIXED_POINT_PRECISION);
                break;
            case Conv2d:
                if (layer->padding == Same){
                    output->numRows = input->numRows / layer->stride_numRows;
                    if (input->numRows % layer->stride_numRows > 0)output->numRows ++;
                    output->numCols = input->numCols / layer->stride_numCols;
                    if (input->numCols % layer->stride_numRows > 0) output->numCols ++;
                }
                else{
                    output->numRows = (input->numRows - layer->kernel->numRows + 1) / layer->stride_numRows;
                    if ((input->numRows - layer->kernel->numRows + 1) % layer->stride_numRows > 0) output->numRows ++;
                    output->numCols = (input->numCols - layer->kernel->numCols + 1) / layer->stride_numCols;
                    if ((input->numCols - layer->kernel->numCols + 1) % layer->stride_numCols > 0) output->numCols ++;
                }
                conv2d(output, input, layer->kernel, layer->numFilters, layer->numChannels, layer->bias->data, layer->activation, FIXED_POINT_PRECISION, layer->stride_numRows, layer->stride_numCols, layer->padding);
                break;
            case Flatten:
                output->numRows = input->numRows * input->numCols * layer->numChannels;
                output->numCols = LEA_RESERVED;
                flatten(output, input, layer->numChannels);
                break;
            case Maxpooling:
                output->numRows = input->numRows / layer->pool_numRows;
                output->numCols = input->numCols / layer->pool_numCols;
                maxpooling_filters(output, input, layer->numChannels, layer->pool_numRows, layer->pool_numCols);
                break;
            default:
                break;
        }
        memset(input->data, 0, sizeof(dtype) * INPUT_OUTPUT_BUFFER_LENGTH);
        memcpy(input->data, output->data, sizeof(dtype) * INPUT_OUTPUT_BUFFER_LENGTH);
        memset(output->data, 0, sizeof(dtype) * INPUT_OUTPUT_BUFFER_LENGTH);
        input->numRows = output->numRows;
        input->numCols = output->numCols;
    }
    memcpy(model->output->data, input->data, sizeof(dtype) * input->numRows * input->numCols);
    return model->output;
}




