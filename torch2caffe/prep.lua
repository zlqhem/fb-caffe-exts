require 'nn'
require 'cunn'
require 'cudnn'

local trans = require 'torch2caffe.transforms'

g_t2c_preprocess = function(model, opts)
  model = cudnn.convert(model, nn)
  --model = trans.fold_batch_normalization_layers(model, opts)
  for _, layer in pairs(model:findModules('nn.SpatialConvolution')) do
    layer.weight = layer.weight:float()
    if layer.bias then
      layer.bias = layer.bias:float()
    end
  end
  for _, layer in pairs(model:findModules('nn.Linear')) do
    layer.weight = layer.weight:float()
    if layer.bias then
      layer.bias = layer.bias:float()
    end
  end
  for _, layer in pairs(model:findModules('nn.SpatialBatchNormalization')) do
    layer.weight = layer.weight:float()
    layer.bias = layer.bias:float()
    layer.running_mean = layer.running_mean:float()
    layer.running_var = layer.running_var:float()
  end
  return model
end

