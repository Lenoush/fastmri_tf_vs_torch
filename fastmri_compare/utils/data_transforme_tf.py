import os
import h5py
from matplotlib import pyplot as plt
import tensorflow as tf

from fastmri_compare.vcr_C.vcr_tf import virtual_coil_reconstruction 

def multicoil_directory_into_singlecoil_directory(input_directory, outuput_directory):
    r""" Search for all .h5 MULTICOIL files in the input_directory and convert them into singlecoil .h5 files
    then deplace it in the output_directory

    Arguments :
        input_directory (str) : path to the directory where all MULTICOIL files are
        outuput_directory (str) : path to the directory where all SINGELCOIL files will be saved
    Returns :
        Message : string
    """

    if not os.path.exists(outuput_directory):
        os.makedirs(outuput_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".h5"):
            with h5py.File(os.path.join(input_directory, file_name), 'r') as hf:
                kspace = hf['kspace'][()]
                image_groundtruth = hf['reconstruction_rss'][()]

            kspace_tensor = tf.convert_to_tensor(kspace, dtype=tf.complex64)

            image = tf.signal.ifft2d(kspace_tensor)
            image_vrc = virtual_coil_reconstruction(image)

            kspace_new_image = tf.signal.fft2d(image_vrc)
            output_file_path = os.path.join(outuput_directory, file_name)
            
            with h5py.File(output_file_path, 'w') as hf_out:
                hf_out.create_dataset("kspace", data=kspace_new_image.numpy())
                hf_out.create_dataset("reconstruction_esc", data=image_vrc)

    return "The conversion is done"

# multicoil_directory_into_singlecoil_directory("/volatile/Lena/Data/brain_multicoil_train/", "/volatile/Lena/Data/singlecoil_tf/train/")


from config import  Data_brain_singlecoil_tf

def find_files_with_invalid_shape(directory):
    invalid_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            filepath = os.path.join(directory, filename)
            try:
                with h5py.File(filepath, "r") as f:
                    kspace_shape = f["kspace"].shape
                    print(kspace_shape)
                    if kspace_shape != (16, 640, 320):
                        print(kspace_shape)
                        invalid_files.append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    return invalid_files


print(find_files_with_invalid_shape(Data_brain_singlecoil_tf))