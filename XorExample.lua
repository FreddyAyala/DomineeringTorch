local function main()
local torch = require 'torch'
local input = torch.Tensor(4,2)
input[1][1]=0
input[1][2]=0
input[2][1]=0
input[2][2]=1
input[3][1]=1
input[3][2]=0
input[4][1]=1
input[4][2]=1
local output = torch.Tensor(4,1)
output[1]=0
output[2]=1
output[3]=1
output[4]=0

require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron
inputs=2; outputs=1;HUs=20; --HU is hidden units
--Define the neural network
mlp:add(nn.Linear(inputs,HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs,outputs))
--Train the neural network
criterion = nn.MSECriterion()
for i =1,1000 do
  for j=1,4 do
  --feed it to the neural network and the criterion
  err=criterion:forward(mlp:forward(input[j]),output[j])
  print(err)
  --train over this example in 3 steps
  --(1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  --(2) accumulate gradicleents
  mlp:backward(input[j],criterion:backward(mlp.output,output[j]))
  --(3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
  end
end
print("Test")
for j=1,4 do
  out = mlp:forward(input[j])
  print("input:")
  print(input[j])
  print("output:")
  print(out[1])
end
end

main()
