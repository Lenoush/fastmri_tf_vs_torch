import time
from matplotlib import pyplot as plt
import os
import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from fastmri_recon.models.functional_models.unet import unet

from fastmri_compare.vcr_C.vcr_tf import virtual_coil_reconstruction 
from config import Data_brain_multicoil, Data_brain_MtoS

def process_h5_files(input_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(input_path):
        if file_name.endswith(".h5"):
            with h5py.File(os.path.join(input_path, file_name), 'r') as hf:
                kspace = hf['kspace'][()]

            kspace_tensor = tf.convert_to_tensor(kspace, dtype=tf.complex64)
            image = tf.signal.ifft2d(kspace_tensor)
            image_vrc = virtual_coil_reconstruction(image)
            kspace_new_image = tf.signal.fft2d(image_vrc)
            output_file_path = os.path.join(output_path, file_name)
            
            with h5py.File(output_file_path, 'w') as hf_out:
                hf_out.create_dataset("kspace", data=kspace_new_image.numpy())
                hf_out.create_dataset("reconstruction_esc", data=kspace_new_image.numpy())

    return output_path

# paths
train_path = Data_brain_multicoil
val_path = Data_brain_multicoil

test = Data_brain_MtoS
New_train_path = process_h5_files(train_path,test)

n_volumes_train = 2

# generators
AF = 4

train_gen = ZeroFilled2DSequence(New_train_path, af=AF, norm=True)
val_gen = ZeroFilled2DSequence(New_train_path, af=AF, norm=True)

run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
    'layers_n_non_lins': 2,
}
n_epochs = 3
run_id = f'unet_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'


chkpt_cback = ModelCheckpoint(chkpt_path, save_freq=100, save_best_only= True)
# log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    # log_dir=log_dir,
    profile_batch=0,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()

model = unet(input_size=(640, 320, 1), lr=1e-3, **run_params)

model.fit(
    train_gen,
    steps_per_epoch=1,
    # steps_per_epoch=n_volumes_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=1,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback],
    max_queue_size=100,
    use_multiprocessing=True,
    workers=35,
)


result = model.predict(train_gen, steps=n_volumes_train)
print(result.shape)
plt.imshow(result[8,:,:,0])
plt.show()