from matplotlib import pyplot as plt
import torch
import h5py

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)


def load_and_transform(path):
    hf = h5py.File(path)
    kspace = hf['kspace'][()]
    kspace = torch.tensor(kspace , dtype=torch.complex64)
    return kspace


def create_zero_filled_reconstruction(mask, kspace):
    masked_data , _ = mask(kspace.shape) 
    masked_kspace = kspace * masked_data
    masked_image = torch.fft.fftshift(torch.fft.ifft2(masked_kspace))

    masked_image = masked_image.unsqueeze(1)
    return masked_image, masked_data

