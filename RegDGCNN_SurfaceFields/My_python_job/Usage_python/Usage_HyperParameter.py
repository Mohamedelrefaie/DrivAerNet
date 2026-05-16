# === Variable  ===
1. batch_size
   -> It defines how many samples are processed at once per training step
   -> e.g. batch_size = 2
      -> each batchcontains 2 samples
   -> In train.py
      -> the 2 samples is .vtk files
      -> each sample have two variable, points_tensor and pressure_tensor

2. epochs
 Total number of training rounds over the whole dataset
 e.g. 10000 samples, batch_size = 100
      10 000 / 100 = 100 batches for one epoch
      epochs 150 have 150 times loop

