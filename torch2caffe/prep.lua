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

local function adapt_sequential_dropout(model)
  -- does not support recursive sequential(dropout)
  for k, block in pairs(model:findModules('nn.SequentialDropout')) do
    -- find last conv / bn / linear layer and scale its weight by 1-p
    local found = false
    for j = #block.modules,1,-1 do
      local block_type = torch.type(block.modules[j])
      if block_type == 'nn.SpatialConvolution'
          or block_type == 'nn.Linear'
          or block_type == 'nn.SpatialBatchNormalization' then
        block.modules[j].weight:mul(1 - block.p)
        if block.modules[j].bias then
          block.modules[j].bias:mul(1 - block.p)
        end
        found = true
        break
      end
    end
    if not found then
      error('SequentialDropout module cannot find weight to scale')
    end
  end
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
  adapt_sequential_dropout(model)
  return model
end

