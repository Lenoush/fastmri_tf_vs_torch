import numpy as np
import torch
import time

from fastmri_compare.utils.other import load_and_transform
from config import Data_brain_multicoil

file_path = Data_brain_multicoil+"file_brain_AXT1POST_201_6002778.h5"
kspace_multicoil = load_and_transform(file_path)
imgs = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))

 # TF
start_tf = time.time()
from vcr_tf import virtual_coil_reconstruction as VCR_tf
tf_output = VCR_tf(imgs)
end_tf = time.time()

# PyTorch
imgs = torch.tensor(imgs)
start_torch = time.time()
from vcr_torch import virtual_coil_reconstruction as VCR_torch
pt_output = VCR_torch(imgs)
end_torch = time.time()


print("tf time :", end_tf - start_tf)
print("torch time : ", end_torch - start_torch)

# Assurez-vous que les formes sont correctes
assert tf_output.shape == pt_output.shape
print("Shapes match.")

# Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)
print("Values are close.")




