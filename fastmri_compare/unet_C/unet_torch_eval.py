import torch
from fastmri.models.unet import Unet
from matplotlib import pyplot as plt 

from fastmri_compare.unet_C.unet_torch_training_script import filename_to_image_and_kspace, get_zerofilled
from config import Save_model_path, Data_brain_multicoil

# Charger le modèle pré-entraîné
model = Unet(in_chans=1, out_chans=1, num_pool_layers=4)
model.load_state_dict(torch.load(Save_model_path+"fastmri_unet_model.pth"))
model.eval()

# Prétraiter vos nouvelles données
filepath = Data_brain_multicoil+'file_brain_AXT1POST_201_6002780.h5'
image, kspace = filename_to_image_and_kspace(filepath)
zero_filled = get_zerofilled(kspace)
zero_filled = torch.abs(zero_filled)

# Effectuer des prédictions
with torch.no_grad():
    output = model(zero_filled)

print(output.shape)
plt.imshow(output[8,0,:,:])
plt.show()
