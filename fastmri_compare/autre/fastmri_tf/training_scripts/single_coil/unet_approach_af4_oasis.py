import sys
sys.path.append('/volatile/Lena/Codes/Modèles/fastmri_tf_vs_torch/fastmri_compare/autre')
print(sys.path)

import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_tf.data.sequences.oasis_sequences import ZeroFilled2DSequence
from fastmri_tf.models.functional_models.unet import unet

# paths
train_path = '/volatile'
# train_path = '/media/Zaccharie/UHRes/OASIS_data/'


# generators
AF = 4
train_gen = ZeroFilled2DSequence(
    train_path,
    af=AF,
    inner_slices=32,
    scale_factor=1e-2,
    seed=0,
    rand=False,
    val_split=0.1,
    n_pooling=3,
)
val_gen = train_gen.val_sequence
n_train = 10 #1000
n_val = 200

run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
    'layers_n_non_lins': 2,
}
n_epochs = 5 #300
run_id = f'unet_af{AF}_oasis_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
print(run_id)




chkpt_cback = ModelCheckpoint(chkpt_path, save_freq=100)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    profile_batch=0,
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()




model = unet(input_size=(None, None, 1), lr=1e-3, **run_params)
print(model.summary())


model.fit(
    train_gen,
    steps_per_epoch=n_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=1,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback],
    # max_queue_size=100,
    use_multiprocessing=True,
    workers=35,
)
