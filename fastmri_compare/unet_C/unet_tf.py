import sys
sys.path.append('/home/lo276838/Modèles/fastmri_tf_vs_torch')

import time 
start = time.time()

import tensorflow as tf

from autre.fastmri_tf.models.functional_models.unet import unet
# from fastmri_compare.vcr_C.vcr_tf import virtual_coil_reconstruction
from helpLena import virtual_coil_reconstruction
from autre.fastmri_tf.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace
from autre.fastmri_tf.data.utils.masking.gen_mask import gen_mask



def filename_to_image_and_kspace(filename) :
    _ , kspace_multi = from_multicoil_train_file_to_image_and_kspace(filename)
    # kspace_multi = tf.convert_to_tensor(kspace_multi)
    image_multi = tf.signal.ifft2d(kspace_multi)
    image = virtual_coil_reconstruction(image_multi)
    kspace = tf.signal.fft2d(image)
    return image, kspace


def get_zerofilled(kspace, af) :
    zero_img_batch = list()
    for batch in range (kspace.shape[0]) :
        mask = gen_mask(kspace[batch], accel_factor=af)
        masked_kspace = kspace[batch] * mask
        zero_filled = tf.signal.ifft2d(masked_kspace)
        zero_filled = tf.expand_dims(zero_filled,0)
        zero_img_batch.append(zero_filled)

    zero_filled = tf.concat(zero_img_batch, axis = 0)
    input = tf.abs(zero_filled)
    return input



# train_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"

# af = 4
# lr = 1e-3
# run_params = {
#     'n_layers': 4,
#     'pool': 'max',
#     "layers_n_channels": [16, 32, 64, 128],
#     'layers_n_non_lins': 2,
# }

# image , kspace = filename_to_image_and_kspace(train_path)
# target = tf.abs(image)

# zero_filled = get_zerofilled(kspace, af=af)

# model = unet(input_size=zero_filled.shape, lr=lr, **run_params)
# criterion = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# loss_list = []
# num_epochs = 2 
# zero_filled = tf.expand_dims(zero_filled, 0)
# for epoch in range(num_epochs):
    
#     with tf.GradientTape() as tape:
#         outputs = model(zero_filled)
#         loss = criterion(outputs, target)

#     loss_list.append(loss)

#     gradients = tape.gradient(loss_list, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     print(f'Epoch [{epoch + 1}/{num_epochs}]')

# end = time.time()
# print( end - start)