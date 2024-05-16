import time

from fastmri_compare.unet_C.ImageSaverCallback import ImageSaverCallback
start = time.time()

from matplotlib import pyplot as plt
import os.path as op
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from fastmri_recon.models.functional_models.unet import unet
from config import Checkpoints_Dir, Data_brain_singlecoil_predict_tf, Data_brain_singlecoil_tf

endimport = time.time()

# paths
train_path = Data_brain_singlecoil_predict_tf
n_volumes_train = len(os.listdir(train_path)) // 16

# generators
AF = 4
train_gen = ZeroFilled2DSequence(train_path, af=AF, norm=True)
val_gen = ZeroFilled2DSequence(train_path, af=AF, norm=True)

run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
}

n_epochs = 3
run_id = f'unet_af{AF}_{int(time.time())}'
chkpt_path = f'{Checkpoints_Dir}unetTF_model.hdf5'

chkpt_cback = ModelCheckpoint(chkpt_path, save_freq=100, save_weights_only=True, verbose=1)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    log_dir=log_dir,
    profile_batch=0,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()
save_path = '/volatile/Lena/Codes/Modèles/fastmri_tf_vs_torch-1/fastmri_compare/unet_C/training_images'
os.makedirs(save_path, exist_ok=True)
image_saver_cb = ImageSaverCallback(val_gen, save_path)

model = unet(input_size=(640, 320, 1), lr=1e-3, **run_params)
model.save_weights(chkpt_path.format(epoch=n_epochs))
# model.load_weights(chkpt_path)
# for layer in model.layers:
#     weights = layer.get_weights() 
#     print(f"Layer: {layer.name}")
#     for weight in weights:
#         print(weight.shape)


history = model.fit(
    train_gen,
    steps_per_epoch=n_volumes_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=0,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback, image_saver_cb],
    max_queue_size=100,
    use_multiprocessing=True,
    workers=35,
)
endfinal = time.time()  

print(f"Time of import : {endimport - start}")
print(f"Time of final training  : {endfinal - endimport}")

history.history

def display_images(epoch, save_path):
    img_path = os.path.join(save_path, f'epoch_{epoch}.png')
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Display images for a specific epoch
epoch_to_display = 3
display_images(epoch_to_display, save_path)