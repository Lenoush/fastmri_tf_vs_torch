import numpy as np
import torch
import time


from unet_torch import filename_to_image_and_kspace, get_zerofilled
from fastmri.models.unet import Unet


file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"



def test_combine_images(filename):

    chans = 32
    num_pool_layers = 4
    lr = 0.0001


    # PyTorch
    start_torch = time.time()

    image, kspace = filename_to_image_and_kspace(filename)
    target = image.abs()

    zero_filled = get_zerofilled(kspace)

    model_torch = Unet(in_chans=1, out_chans=1, chans=chans, num_pool_layers=num_pool_layers)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model_torch.parameters(), lr=lr)

    loss_list = []
    num_epochs = 2 # test on 200 , from 54 not much change

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        outputs = model_torch(zero_filled)
        loss = criterion(outputs, target)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]')


    end_torch = time.time()



    # # TF
    # start_tf = time.time()
    # tf_output = VCR_tf(download_data)
    # end_tf = time.time()

    # # Assurez-vous que les formes sont correctes
    # assert tf_output.shape == pt_output.shape
    # print("Shapes match.")

    # # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    # np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)
    # print("Values are close.")

    # print("tf time :", end_tf - start_tf)
    print("torch time : ", end_torch - start_torch)

test_combine_images(file_path)

