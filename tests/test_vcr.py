
from fastmri_compare.vcr_C.vcr_tf import virtual_coil_reconstruction as VCR_tf
from fastmri_compare.vcr_C.vcr_torch import virtual_coil_reconstruction as VCR_torch
from fastmri_compare.utils.other import load_and_transform
from config import Data_brain_multicoil

import numpy as np
import pytest
import torch


@pytest.fixture
def download_data() :
    file_path = Data_brain_multicoil+"file_brain_AXT1POST_201_6002780.h5"
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
    np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=8)
