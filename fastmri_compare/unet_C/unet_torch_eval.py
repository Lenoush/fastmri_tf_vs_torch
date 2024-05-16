import time 

strat = time.time()

import torch
import os
from fastmri.models.unet import Unet
from matplotlib import pyplot as plt 

from fastmri_compare.unet_C.unet_torch_training_script import get_zerofilled
from fastmri_compare.utils.data_transforme import path_to_image_and_kspace
from config import Save_model_path, Data_brain_singlecoil_predict_pytorch

endimport = time.time()

# Charger le modèle pré-entraîné
model = Unet(in_chans=1, out_chans=1, num_pool_layers=4)
model.load_state_dict(torch.load(Save_model_path+"fastmriTorch_unet_model.pth"))
model.eval()

# Prétraiter vos nouvelles données
filepath = Data_brain_singlecoil_predict_pytorch
for file_name in os.listdir(filepath):
    if file_name.endswith(".h5"):
        image, kspace = path_to_image_and_kspace(os.path.join(filepath, file_name))

plt.imshow(abs(image[8,0,:,:]), cmap="gray")
plt.title("Image (Target)")
plt.show()

zero_filled = get_zerofilled(kspace)
plt.imshow(abs(zero_filled[8,0,:,:]), cmap="gray")
plt.title("Zero-filled reconstruction")
plt.show()

# Effectuer des prédictions
with torch.no_grad():
    output = model(abs(zero_filled))
    loss = torch.nn.functional.l1_loss(output, image)
    print(f'Loss: {loss.item()}')

endend = time.time()

print(f"Time of import: {endimport - strat}")
print(f"Time of final: {endend - endimport}")

plt.imshow(output[8,0,:,:], cmap='gray')
plt.title("Prediction (UNET)")
plt.show()
