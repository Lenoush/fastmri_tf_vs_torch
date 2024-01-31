import sys
sys.path.append('/home/lo276838/Modèles/fastmri_tf_vs_torch/fastmri_compare')


import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable

from vcr_C.vcr_torch import virtual_coil_reconstruction
from utils.other import load_and_transform

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models.unet import Unet


def filename_to_image_and_kspace(train_path):
    kspace_multicoil = load_and_transform(train_path)
    images_multicoil = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))
    image = virtual_coil_reconstruction(images_multicoil)
    image = image.unsqueeze(1)
    kspace = torch.fft.fft2(image)

    return image, kspace

def get_zerofilled( 
        kspace,
        mask_type = "random", 
        center_fractions = [0.8],
        accelerations = [4],
        ):
    
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    if len(kspace) == 4 :
        kspace = kspace.unsqueeze(-1)

    zero_filled_list = []
    for batch in range (kspace.shape[0]) :
        mask, _ = mask_func(kspace.shape)
        masked_kspace = kspace[batch] * mask
        zero_filled =torch.fft.ifftn(masked_kspace)
        zero_filled_list.append(zero_filled)

    zero_filled = torch.cat(zero_filled_list)
    zero_filled = zero_filled.squeeze(-1)
    input = np.abs(zero_filled)

    return input 


def train_mulicoil_data(filepath, chans = 32, num_pool_layers=4, lr= 1e-3, num_epochs= 2, name_for_save = 'fastmri_unet_model.pth'):
    image, kspace = filename_to_image_and_kspace(filepath)
    target = image.abs()

    zero_filled = get_zerofilled(kspace)

    model = Unet(in_chans=1, out_chans=1, chans=chans, num_pool_layers=num_pool_layers)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        outputs = model(zero_filled)
        loss = criterion(outputs, target)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]')

    torch.save(model.state_dict(), name_for_save)
    return outputs


    

# filepath = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
# x = train_mulicoil_data(filepath)
# print(x)
