import sys
sys.path.append('/home/lo276838/Modèles/fastmri_tf_vs_torch/fastmri_compare')

from vcr_tf import virtual_coil_reconstruction as VCR_tf
from vcr_torch import virtual_coil_reconstruction as VCR_torch

from ..utils.other import load_and_transform

import numpy as np
import torch
import time



file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
kspace_multicoil = load_and_transform(file_path)
imgs = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))

# imgs = np.random.randn(16, 16, 640, 320,2).astype(np.complex64)

print(imgs.shape)
def test_combine_images(download_data):

    # PyTorch
    download_data= torch.tensor(download_data)
    start_torch = time.time()
    pt_output = VCR_torch(download_data)
    end_torch = time.time()

    # TF
    start_tf = time.time()
    tf_output = VCR_tf(download_data)
    end_tf = time.time()

    # Assurez-vous que les formes sont correctes
    assert tf_output.shape == pt_output.shape
    print("Shapes match.")

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)
    print("Values are close.")

    print("tf time :", end_tf - start_tf)
    print("torch time : ", end_torch - start_torch)

test_combine_images(imgs)

