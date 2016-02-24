function createModel(nGPU)
   local features = nn.Sequential()

   -- 1
   features:add(nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
   features:add(nn.SpatialBatchNormalization(64, 1e-3))
   features:add(nn.SpatialMaxPooling(3, 3, 2, 2))
   features:add(nn.ReLU(true))

   -- 2
   features:add(nn.SpatialConvolution(64, 192, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(192, 192, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   features:add(nn.ReLU(true))

   -- 3
   features:add(nn.SpatialConvolution(192, 192, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(384, 1e-3))
   features:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   features:add(nn.ReLU(true))

   -- 4
   features:add(nn.SpatialConvolution(384, 384, 1, 1))
   features:add(nn.SpatialBatchNormalization(384, 1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(nn.ReLU(true))

   -- 5
   features:add(nn.SpatialConvolution(256, 256, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(nn.ReLU(true))

   -- 6
   features:add(nn.SpatialConvolution(256, 256, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   features:add(nn.ReLU(true))

   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*7*7))

   classifier:add(nn.Linear(256*7*7, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(4096, 128))
   classifier:add(nn.Normalize(2))

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 224
   model.imageCrop = 224

   return model
end
