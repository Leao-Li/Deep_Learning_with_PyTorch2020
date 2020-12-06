# 4.2 3D images: Volumetric data 

import torch
import imageio

dir_path = "E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, "DICOM")

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

vol.shape
