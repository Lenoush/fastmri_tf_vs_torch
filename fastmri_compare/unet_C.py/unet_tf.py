
# import os.path as op
# import time
# import h5py

# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
# from tensorflow_addons.callbacks import TQDMProgressBar

# from fastmri_compare.fastmri_tf.data.sequences.fastmri_sequences import ZeroFilled2DSequence
# from fastmri_compare.fastmri_tf.models.functional_models.unet import unet
# from fastmri_compare.fastmri_tf.data.utils.vcr_tf import virtual_coil_reconstruction 


# import tensorflow as tf
# from tensorflow.python.keras.optimizers import adam_v2
# from tensorflow.python.keras.losses import MeanAbsoluteError

# def load_and_transform(path):
#     hf = h5py.File(path)
#     kspace = hf['kspace'][()]
#     kspace = tf.cast(kspace, dtype=tf.complex64)
#     return kspace


# # paths
# train_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"

# # multi to single coils 
# kspace_multicoil = load_and_transform(train_path)
# images_multicoil = tf.signal.fftshift(tf.signal.ifft2d(kspace_multicoil))
# image = virtual_coil_reconstruction(images_multicoil)
# image = image.unsqueeze(1)
# kspace = tf.signal.fft2d(image)
# target = image.abs()

# new_train_path = "/home/lo276838/Modèles/fastmri_tf_vs_torch/fastmri_compare/compare/unet.py"
# with h5py.File(new_train_path, "w") as f :
#     f.create_group('kspace')
#     f['kspace'].create_dataset('data', data=kspace)


# n_volumes_train = 973

# # generators
# AF = 4 # af (int): the acceleration factor.
# train_gen = ZeroFilled2DSequence(new_train_path, af=AF, norm=True)


# run_params = {
#     'n_layers': 4,
#     'pool': 'max',
#     "layers_n_channels": [16, 32, 64, 128],
#     'layers_n_non_lins': 2,
# }
# run_id = f'unet_af{AF}_{int(time.time())}'
# chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'



# chkpt_cback = ModelCheckpoint(chkpt_path, period=100)
# log_dir = op.join('logs', run_id)
# tboard_cback = TensorBoard(
#     log_dir=log_dir,
#     profile_batch=0,
#     histogram_freq=0,
#     write_graph=True,
#     write_images=False,
# )
# tqdm_cb = TQDMProgressBar()


# model = unet(input_size=(16,640, 320), lr=1e-3, **run_params)
# # print(model.summary())

# criterion = MeanAbsoluteError()
# optimizer = adam_v2.Adam(learning_rate=1e-3)

# loss_list = []
# n_epochs = 2 
# print("essaie")
# for epoch in range(n_epochs):
#     input = tf.Variable(input)
#     target = tf.Variable(target)

#     optimizer.zero_grad()
#     outputs = model(input)
#     loss = criterion(outputs, target)
#     loss_list.append(loss.item())

#     loss.backward()
#     optimizer.step()

#     print(f'Epoch [{epoch + 1}/{n_epochs}]')

# # Save the model
# tf.saved_model.save(model, 'fastmri_unet_model')



# # model.fit_generator(
# #     train_gen,
# #     steps_per_epoch=n_volumes_train,
# #     epochs=n_epochs,
# #     validation_data=val_gen,
# #     validation_steps=1,
# #     verbose=0,
# #     callbacks=[tqdm_cb, tboard_cback, chkpt_cback],
# #     # max_queue_size=100,
# #     use_multiprocessing=True,
# #     workers=35,
# # )


import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from fastmri_recon.models.functional_models.unet import unet





# paths
train_path = '/media/Zaccharie/UHRes/singlecoil_train/singlecoil_train/'
val_path = '/media/Zaccharie/UHRes/singlecoil_val/'
test_path = '/media/Zaccharie/UHRes/singlecoil_test/'





n_samples_train = 34742
n_samples_val = 7135

n_volumes_train = 973
n_volumes_val = 199





# generators
AF = 4
train_gen = ZeroFilled2DSequence(train_path, af=AF, norm=True)
val_gen = ZeroFilled2DSequence(val_path, af=AF, norm=True)





run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
    'layers_n_non_lins': 2,
}
n_epochs = 300
run_id = f'unet_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'





chkpt_cback = ModelCheckpoint(chkpt_path, period=100)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    log_dir=log_dir,
    profile_batch=0,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()




model = unet(input_size=(320, 320, 1), lr=1e-3, **run_params)
print(model.summary())





model.fit_generator(
    train_gen,
    steps_per_epoch=n_volumes_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=0,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback],
    # max_queue_size=100,
    use_multiprocessing=True,
    workers=35,
)
