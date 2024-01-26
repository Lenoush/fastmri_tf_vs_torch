
import sys
sys.path.append('/home/lo276838/Modèles/fastmri_tf_vs_torch/fastmri_compare')

from fastmri_compare.fastmri_tf.data.utils.vcr_tf import virtual_coil_reconstruction as VCR_tf
from fastmri_compare.fastmri_torch.models.utils.utils_torch import virtual_coil_reconstruction as VCR_torch
from fastmri_compare.fastmri_torch.models.utils.utils_torch import load_and_transform


import numpy as np
import pytest
import torch


@pytest.fixture
def download_data() :
    file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
    kspace_multicoil = load_and_transform(file_path)
    images_multicoil = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))

    return images_multicoil


def test_combine_images(download_data):

    # PyTorch
    pt_output = VCR_torch(download_data)

    # TF
    tf_output = VCR_tf(download_data)


    # Assurez-vous que les formes sont correctes
    assert tf_output.shape == pt_output.shape

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)
