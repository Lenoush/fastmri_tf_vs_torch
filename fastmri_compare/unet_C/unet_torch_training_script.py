import torch
import glob

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models.unet import Unet

from fastmri_compare.vcr_C.vcr_torch import virtual_coil_reconstruction
from fastmri_compare.utils.other import load_and_transform
from config import Save_model_path, Data_brain_multicoil

def filename_to_image_and_kspace(train_path):
    r""" Load and transform Multi-coil into Single-coil.

    Arguments :
        train_path (str) : path of the data with the extension .h5
    Returns :
        image : image of all batchs 
        kspace : kspace of all batchs
    """

    kspace_multicoil = load_and_transform(train_path)
    images_multicoil = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))
    image = virtual_coil_reconstruction(images_multicoil)
    image = image.unsqueeze(1)
    kspace = torch.fft.fft2(image)

    return image, kspace


def get_zerofilled( 
        kspace,
        mask_type = "random", 
        center_fractions = [0.08],
        accelerations = [4],
        ):
    r""" Create masks for each batchs and apply it by multiplying the mask and the batch.

    Arguments :
        kspace (array) : kspace of my data. Shape : (Batch, Coils = 1, H, W)
            TEST FOR DIM  = 3 !!
        mask_type (str) : choose the mask func for the target mask type
            Default : "random"
        center_fractions (array) : What fraction of the center of k-space to include.
            Default : [0.8]
        accelerations (array) : What accelerations to apply.
            Default : [4]
    Returns :
        zero_filled : the reconstructed masked kspaces
    """
    
    mask_func = create_mask_for_mask_type(
        mask_type, 
        center_fractions, 
        accelerations)

    if len(kspace.shape) == 4 :
        kspace = kspace.unsqueeze(-1)
    
    zero_filled_list = []
    for batch in range (kspace.shape[0]) :
        mask, _ = mask_func(kspace.shape)
        masked_kspace = kspace[batch] * mask
        zero_filled =torch.fft.ifftn(masked_kspace)
        zero_filled_list.append(zero_filled)

    zero_filled = torch.cat(zero_filled_list)
    zero_filled = zero_filled.squeeze(-1)

    return zero_filled


def train_mulicoil_data(
        filespath, 
        num_epochs = 200,
        num_pool_layers=4, 
        lr=1e-3, 
        name_for_save= Save_model_path+'fastmri_unet_model.pth'
        ):
    r""" Train an Unet network on the fastMRI dataset.

    Arguments :
        filespath (str) : path to the directory where all datas are. In this dirctory, it have to have at least one .h5 file.
        num_epochs (int) : the number of epochs (i.e. one pass though all the volumes/samples) for this training.
            Default : 200
        num_pool_layers (int) : Number of down-sampling and up-sampling layers.
            Default : 4
        lr (float) : learning rate. 
            Default : 1e-3
        name_for_save : Name with full path indicating where to save the model.
    Returns :
        outputs10 (array) :  output list every 10 epochs
    """

    filenames = glob.glob(filespath + '*.h5')
    if not filenames:
        raise ValueError('No h5 files at path {}'.format(filespath))

    model = Unet(in_chans=1, out_chans=1, num_pool_layers=num_pool_layers)
    model.train()

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    outputs10 = []

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        i = 0

        for filename in filenames:
            print(f'FileNumber [{i+1}/{len(filenames)}]')

            image, kspace = filename_to_image_and_kspace(filename)
            target = image.abs()
            zero_filled = get_zerofilled(kspace)
            zero_filled = torch.abs(zero_filled)

            optimizer.zero_grad()
            outputs = model(zero_filled)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())
            if epoch % 10 == 0:
                outputs10.append([outputs, loss.item()])
            loss.backward()
            optimizer.step()
            i+=1

    torch.save(model.state_dict(), name_for_save)
    return outputs10


if __name__ == '__main__':
    filepath = Data_brain_multicoil
    x = train_mulicoil_data(filepath, num_epochs=1)