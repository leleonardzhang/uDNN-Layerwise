#include "layers.h"
#include "neural_network_parameters.h"
#include "math/matrix_ops.h"
#include "math/fixed_point_ops.h"
#include "math/matrix.h"
#include "utils/utils.h"
#include "layer.h"

#ifndef LAYERS_PREDICT_H_
#define LAYERS_PREDICT_H_

matrix *predict(model_t *model);

#endif /* LAYERS_PREDICT_H_ */
