require 'nn'
require 'cunn'
require 'cudnn'

local trans = require 'torch2caffe.transforms'

local function adapt_conv1(layer)
  local std = torch.FloatTensor({0.229, 0.224, 0.225}) * 255
  local sz = layer.weight:size()
  sz[2] = 1
  layer.weight = layer.weight:cdiv(std:view(1,3,1,1):repeatTensor(sz))
  local tmp = layer.weight:clone()
  tmp[{{}, 1, {}, {}}] = layer.weight[{{}, 3, {}, {}}]
  tmp[{{}, 3, {}, {}}] = layer.weight[{{}, 1, {}, {}}]
  layer.weight = tmp:clone()
end

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
  adapt_conv1(model.modules[1])
  return model
end

