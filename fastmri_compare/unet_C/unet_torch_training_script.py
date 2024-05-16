import time

start = time.time()

import torch
import glob
import matplotlib.pyplot as plt

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri_compare.utils.data_transforme import path_to_image_and_kspace
from fastmri.models.unet import Unet

from config import Save_model_path, Data_brain_singlecoil_pytorch, Data_brain_singlecoil_predict_pytorch

endimport = time.time()

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
        mask_type (str) : choose the mask func for the zero-filled mask type
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


def train_data(
        filespath, 
        num_epochs = 200,
        num_pool_layers=3, 
        lr=1e-3, 
        name_for_save= Save_model_path+'fastmriTorch_unet_model.pth'
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
    # Data
    filenames = glob.glob(filespath + '*.h5')
    if not filenames:
        raise ValueError('No h5 files at path {}'.format(filespath))
    
    # Model
    model = Unet(in_chans=1, out_chans=1, chans=16, num_pool_layers=num_pool_layers)
    model.train()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    outputs_and_loss_by_epoch = []
    traindataset = []

    # Create the dataset
    for filename in filenames:
        image, kspace = path_to_image_and_kspace(filename)
        # plt.imshow(abs(image[8,0,:,:]), cmap="gray")
        # plt.title("Image (Target)")
        # plt.show()

        target = image.abs()
        zero_filled = get_zerofilled(kspace)
        # plt.imshow(abs(zero_filled[8,0,:,:]), cmap="gray")
        # plt.title("Zero-filled reconstruction")
        # plt.show()

        traindataset.append([zero_filled, target])

    # Train the model
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')

        i = 0
        result_by_epoch = []

        for zero_filled, target in traindataset:
            print(f'Image [{i+1}/{len(traindataset)}]')

            # Forward pass
            optimizer.zero_grad()
            outputs = model(abs(zero_filled))
            plt.imshow(outputs[8,0,:,:].detach().numpy(), cmap="gray")
            plt.title("Prediction (UNET)")
            plt.show()

            # Save the output and loss
            loss = criterion(outputs, target)
            result_by_epoch.append([outputs, loss.item()])
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            i+=1

        outputs_and_loss_by_epoch.append(result_by_epoch)

    # Save the model 
    # torch.save(model.state_dict(), name_for_save)


    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    return outputs_and_loss_by_epoch

if __name__ == '__main__':
    filepath = Data_brain_singlecoil_predict_pytorch 
    trainning_result = train_data(filepath, num_epochs=3)
    endfinal = time.time()

    # Plot the loss for each epoch 
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for epoch in range(len(trainning_result)):
    #     index_image = [image for image in range(len(trainning_result[epoch]))] 
    #     lost_by_image = [loss[1] for loss in trainning_result[epoch]]
    #     plt.plot(index_image, lost_by_image, color=colors[epoch % len(colors)], label=f'Epoch {epoch+1}')

    # plt.title('Loss for each image for each epoch')
    # plt.xlabel('Image')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    print(f"Time of import : {endimport - start}")
    print(f"Time of final : {endfinal - endimport}")
    print(f"Time of all : {endfinal - start}")