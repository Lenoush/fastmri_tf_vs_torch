import torch
import glob 
from matplotlib import pyplot as plt 

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models.varnet import VarNet 
from config import Save_model_path, Data_brain_multicoil
from fastmri_compare.unet_C.unet_torch_training_script import filename_to_image_and_kspace

def get_masked_kspace( 
        kspace,
        mask_type = "equispaced", 
        center_fractions = [0.8],
        accelerations = [4],
        ):
    
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    if (len(kspace.shape) == 4 ):
        kspace = kspace.unsqueeze(-1)

    masked_kspace_list, mask_list = [], []
    for batch in range (kspace.shape[0]) :
        mask, _ = mask_func(kspace.shape)
        masked_kspace = kspace[batch] * mask

        masked_kspace_list.append(masked_kspace)
        mask_list.append(mask)

    masked_kspace = torch.cat(masked_kspace_list)
    mask = torch.cat(mask_list)

    return masked_kspace, mask


def BonneShape(image, masked_kspace) :
    masked_kspace = masked_kspace.squeeze(-1)
    # Création d'une nouvelle dimension pour dissocier la partie réel et imaginaire des complex
    shape_new = image.shape + (2,)
    masked_kspace_new = torch.zeros(shape_new)
    masked_kspace_new[..., 0] = masked_kspace.real
    masked_kspace_new[..., 1] = masked_kspace.imag 

    return masked_kspace_new

def train_Varnet_multicoil_data(filepath, num_epochs, num_pool_layers=4, lr=1e-3, name_for_save=Save_model_path+'fastmri_varnet_model.pth'):

    filenames = glob.glob(filepath + '*.h5')
    if not filenames:
        raise ValueError('No h5 files at path {}'.format(filepath))

    model = VarNet(
        num_cascades=2,
        sens_chans=4,
        sens_pools=2,
        chans=32,
        pools=2,
        mask_center=True,
    )
    model.train()

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    outputs10 = []

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        i = 0

        for filename in filenames:
            image, kspace = filename_to_image_and_kspace(filename)
            target = image.abs().squeeze(1)

            masked_kspace, mask = get_masked_kspace(kspace)
            masked_kspace = BonneShape(image, masked_kspace) 
            
            optimizer.zero_grad()
            outputs = model(masked_kspace, mask.byte())
            outputs = torch.fft.fftshift(outputs)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())
            if epoch % 10 == 0:
                outputs10.append([outputs, loss.item()])
            loss.backward()
            optimizer.step()
            i+=1

    torch.save(model.state_dict(), name_for_save)
    return outputs10


# if __name__ == '__main__':
#     filepath = Data_brain_multicoil
#     x = train_Varnet_multicoil_data(filepath, num_epochs=1)
#     print(x[0][0].shape)
#     plt.imshow(x[0][0][8,:,:].detach().numpy())
#     plt.show()