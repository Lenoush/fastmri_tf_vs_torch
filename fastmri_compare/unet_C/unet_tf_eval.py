import time

start = time.time()
from matplotlib import pyplot as plt
from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence
from fastmri_recon.models.functional_models.unet import unet
from config import Data_brain_singlecoil_predict_tf, Checkpoints_Dir

endimport = time.time()

run_params = {
'n_layers': 4,
'pool': 'max',
"layers_n_channels": [16, 32, 64, 128],
'layers_n_non_lins': 2,
}

model = unet(input_size=(640, 320, 1), lr=1e-3, **run_params)

endrunmode = time.time()

predict_zerofilled = ZeroFilled2DSequence(Data_brain_singlecoil_predict_tf, af=4, norm=True)

# Load the model
history = model.load_weights(f'{Checkpoints_Dir}unetTF_model.hdf5')
result = model.predict(predict_zerofilled, steps=1)
endpredict =time.time()

loss = model.evaluate(predict_zerofilled, steps=1)
print(model.metrics_names)
print(f'Loss: {loss}')

print(f"Time of import : {endimport - start}")
print(f"Time of model : {endrunmode - endimport}")
print(f"Time of predict : {endpredict - endrunmode}")

print(result.shape)
plt.imshow(result[8,:,:,0], cmap='gray')
plt.title('image predict')
plt.show()

# import os
# from matplotlib import pyplot as plt
# save_path = '/volatile/Lena/Codes/Modèles/fastmri_tf_vs_torch-1/fastmri_compare/unet_C/training_images'

# def display_images(epoch, save_path):
#     img_path = os.path.join(save_path, f'epoch_{epoch}.png')
#     img = plt.imread(img_path)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

# # Display images for a specific epoch
# epoch_to_display = 1
# display_images(epoch_to_display, save_path)