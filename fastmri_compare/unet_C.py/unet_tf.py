import sys
sys.path.append('/home/lo276838/Modèles/fastmri_tf_vs_torch')

# import os.path as op
import time
import tensorflow as tf
import numpy as np

# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
# from tensorflow_addons.callbacks import TQDMProgressBar

# from autre.fastmri_tf.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from autre.fastmri_tf.models.functional_models.unet import unet
# from fastmri_compare.vcr_C.vcr_tf import virtual_coil_reconstruction 

from autre.fastmri_tf.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace
from autre.fastmri_tf.evaluate.reconstruction.zero_filled_reconstruction import zero_filled_cropped_recon
from autre.fastmri_tf.data.utils.masking.gen_mask import gen_mask




train_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"

af = 4
lr = 1e-3

image , kspace = from_multicoil_train_file_to_image_and_kspace(train_path)
target = image.abs()

mask = gen_mask(kspace[0], accel_factor=af)
fourier_mask = np.repeat(mask.astype(np.float), kspace[0].shape[0], axis=0)

img_batch = list()
zero_img_batch = list()

for kspace, image in zip(kspace, image):
    zero_filled_rec = zero_filled_cropped_recon(kspace * fourier_mask)
    zero_filled_rec = zero_filled_rec[:, :, None]
    zero_img_batch.append(zero_filled_rec)
    image = image[..., None]
    img_batch.append(image)

zero_filled = np.array(zero_img_batch)
img_batch = np.array(img_batch)


run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
    'layers_n_non_lins': 2,
}




model = unet(input_size=zero_filled.shape, lr=lr, **run_params)


criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

loss_list = []
num_epochs = 2 # 300

for epoch in range(num_epochs):

    optimizer.zero_grad()
    outputs = model(zero_filled)
    loss = criterion(outputs, target)
    loss_list.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
