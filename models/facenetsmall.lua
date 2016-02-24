function createModel(nGPU)
   local features = nn.Sequential()
   features:add(nn.SpatialConvolution(3,48,9,9,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(48,1e-3))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(128,1e-3))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(192,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   features:add(nn.ReLU(true))

   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   classifier:add(nn.Linear(256*6*6, 1024))
   classifier:add(nn.BatchNormalization(1024, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(1024, 1024))
   classifier:add(nn.BatchNormalization(1024, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(1024, 128))
   classifier:add(nn.Normalize(2))

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 224
   model.imageCrop = 224

   return model
end
