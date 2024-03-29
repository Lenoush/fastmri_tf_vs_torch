import torch
from matplotlib import pyplot as plt 
from fastmri.models.varnet import VarNet

from config import Save_model_path, Data_brain_multicoil
from fastmri_compare.unet_C.unet_torch_training_script import filename_to_image_and_kspace
from fastmri_compare.varnetC.varnet_trainning_script import get_masked_kspace, BonneShape

# Charger le modèle pré-entraîné
model = VarNet(
        num_cascades=2,
        sens_chans=4,
        sens_pools=2,
        chans=32,
        pools=2,
        mask_center=True,
    )
model.load_state_dict(torch.load(Save_model_path+"fastmri_varnet_model.pth"))
model.eval()

# Prétraiter vos nouvelles données
filepath = Data_brain_multicoil+'file_brain_AXT1POST_201_6002780.h5'
image, kspace = filename_to_image_and_kspace(filepath)

masked_kspace, mask = get_masked_kspace(kspace)
masked_kspace = BonneShape(image, masked_kspace) 

# Effectuer des prédictions
with torch.no_grad():
    output = model(masked_kspace, mask.byte())
    output = torch.fft.fftshift(output)


print(output.shape)
plt.imshow(output[13,:,:])
plt.show()

