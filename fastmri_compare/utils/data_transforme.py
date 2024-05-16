
import torch
import os
import h5py

from fastmri_compare.vcr_C.vcr_torch import virtual_coil_reconstruction
from fastmri_compare.utils.other import load_and_transform


def path_multicoil_into_singlecoil_image_and_kspace(train_path):
    r""" Load and transform Multi-coil into Single-coil.

    Arguments :
        train_path (str) : PATH of the data MULTICOIL with the extension .h5
    Returns :
        image : image of all batchs 
        kspace : kspace of all batchs
    """
    kspace_multicoil, images_multicoil = load_and_transform(train_path)

    image = virtual_coil_reconstruction(images_multicoil)
    image = image.unsqueeze(1)

    kspace = torch.fft.fft2(image)

    return image, kspace



def path_to_image_and_kspace(train_path):
    r""" Load and transform a path into kSpace and Image

    Arguments :
        train_path (str) : path of the data with the extension .h5
    Returns :
        image : image of all batchs 
        kspace : kspace of all batchs
    """
    kspace = load_and_transform(train_path)
    image = torch.fft.ifft2(kspace)

    return image, kspace


def directory_verif_shape_singlecoil_Torch(path):
    r""" Detect if the shape of the data is correct for the singlecoil data

    Arguments :
        train_path (str) : PATH of the data SINGLECOIL with the extension .h5
    Returns :
        isGoodShape : boolean
    """
    isGodShape = False
    for file_name in os.listdir(path):
        if file_name.endswith(".h5"):
            image , _ = path_to_image_and_kspace(os.path.join(path, file_name))
            if (image.shape == torch.Size([16, 1, 640, 320])) :
                isGodShape = True
    return isGodShape

def path_multicoil_to_singlecoil_directory(multicoil_directory, filename, singlecoil_directory):
    r""" Transform a multicoil file into a singlecoil file and save it in the singlecoil directory

    Arguments :
        multicoil_directory : directory of the multicoil data end with / 
        filename : name of the file with the extension .h5
        singlecoil_directory : directory of the singlecoil data end with /
    Returns :
        Message : str
    """
    full_path_name_multicoil = os.path.join(multicoil_directory, filename)
    full_path_name_singlecoil = os.path.join(singlecoil_directory, filename)

    if os.path.exists(full_path_name_multicoil):
        if full_path_name_multicoil.endswith(".h5"):

            image, kspace = path_multicoil_into_singlecoil_image_and_kspace(full_path_name_multicoil)

            with h5py.File(full_path_name_singlecoil, 'w') as hf_out:
                hf_out.create_dataset("kspace", data=kspace.numpy())
                hf_out.create_dataset("reconstruction_esc", data=image.numpy())

    return "The file has been transformed and saved in the singlecoil directory"


def directory_multicoil_to_singlecoil_directory(multicoil_directory, singlecoil_directory):
    r""" Transform all multicoils files into a singlecoil file and save it in the singlecoil directory

    Arguments :
        multicoil_directory : directory of the multicoil data end with /
        singlecoil_directory : directory of the singlecoil data end with /
    Returns :
        Message : str
    """
    for filename in os.listdir(multicoil_directory):
        if filename.endswith(".h5"):
            path_multicoil_to_singlecoil_directory(multicoil_directory, filename, singlecoil_directory)

    return "All the files have been transformed and saved in the singlecoil directory"

