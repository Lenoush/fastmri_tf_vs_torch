import time
from matplotlib import pyplot as plt 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from fastmri_recon.models.functional_models.unet import unet

from config import Data_brain_singlecoil

# paths
train_path = Data_brain_singlecoil
val_path = Data_brain_singlecoil

n_volumes_train =2 # 973 

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
n_epochs = 2 # 300
run_id = f'unet_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'

chkpt_cback = ModelCheckpoint(chkpt_path, save_freq=100, save_best_only= True) 
tboard_cback = TensorBoard(
    profile_batch=0,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()

model = unet(input_size=(320, 320,1), lr=1e-3, **run_params)
model.fit(
    train_gen,
    steps_per_epoch=n_volumes_train,
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
plt.imshow(result[4,:,:,0])
plt.show()

print(result.shape)
