local function main()
--Function to split the csv file
local function split(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end
local filename = '/home/fayalah/Downloads/org.eclipse.ldt.product-linux.gtk.x86_64/workspace/DomineeringMonteCarlo/src/Domineering2.csv';
local csvFile = io.open(filename ,'r')  
local header = csvFile:read()
local ctr = 0
for _ in io.lines(filename)  do
  ctr = ctr + 1
end
--ctr = 100
local input = torch.Tensor(ctr,3,8,8)
local output = torch.Tensor(ctr,1,8,8)

local linenumber = 0
--For each line in the csv file 
for line in csvFile:lines('*l') do 
if(linenumber>=ctr)
then
  break
 end
--increase the line number 
 linenumber=linenumber+1;
 --set the first tensor number
 tensornumber =1;
  --split the values in the csv
  local l = split(line,',')
  local row =1;
  local column =1
  local i = 0;
    -- for each value in the line   
      for key,val in ipairs(l) do
      --if we have finished procesing the row we get to the next one
     i=i+1
    if(row==8 and column==9)
           then
           row=0 
           end         
   if column == 9
          then 
            column=1
            row=row+1            
        end
        --Tensor 1 Range
        if i >=1 and i<= 64
        then
          tensornumber=1
        end     
        --Tensor 2 RAnge
        if i >= 65 and i<= 128
        then
          tensornumber=2
        end
        --TEnsor 3 RAnge
        if i >=129 and i<= 192
        then
          tensornumber=3
        end
        if i>192
         then
           output[linenumber][1][row][column] = val
        else
           input[linenumber][tensornumber][row][column] = val
        end
         --print("i ".. tostring(i))   
         --print("line number:"..linenumber)
        column=column+1        
      end
    end
  tensornumber =  tensornumber +1
csvFile:close()


--Once we load the csv we can proceed to define the neural network and train it
require 'nn'
require 'optim'
local convnet = nn.Sequential()
local nplanes = 64
--Filter, stride, padding
convnet:add(nn.SpatialConvolution(3,nplanes,5,5,1,1,2,2))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,nplanes,3,3,1,1,1,1))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,nplanes,3,3,1,1,1,1))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,nplanes,3,3,1,1,1,1))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,nplanes,3,3,1,1,1,1))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,nplanes,3,3,1,1,1,1))
convnet:add(nn.ReLU())
convnet:add(nn.SpatialConvolution(nplanes,1,3,3,1,1,1,1))
convnet:add(nn.View(64))
convnet:add(nn.SoftMax())
convnet:add(nn.View(8,8))



--Train the neural network
criterion = nn.MSECriterion()
for j =1,1000 do
  for i =1,ctr-1 do
  print("Cycle:"..j)
  print("Tensor["..i.."]")
    local optimState = {learningRate=0.2}     
    params, gradParams = convnet:getParameters()
       local function feval(params)  
         --print(params[0])
         --gradParams = params[0]:getParameters()
         gradParams:zero()
         local outputs = convnet:forward(input[i])
         local loss = criterion:forward(outputs,output[i])
         local dloss_doutput = criterion:backward(outputs, output[i])
          print('loss:'..loss)   
         convnet:backward(input[i],dloss_doutput )
         return loss, gradParams
       end
 optim.sgd(feval, params, optimState)  
       
   end

  end
--end
  print("Output")
for j=1,ctr-1 do
print(j)
  out = convnet:forward(input[j])
  print("input:")
  print(input[j])
  print("output:(original)")
  print(output[j])
  print("output:(learned)")
  print(out)
  
end
torch.save('domineering.net',convnet)
end

main()
