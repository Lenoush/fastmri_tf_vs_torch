

import time


def test_combine_images(filename):

    #Parametres
    chans = 32
    num_pool_layers = 4
    lr = 1e-3
    accelerations = 4 
    num_epochs = 2
    run_params = {
        'n_layers': 4,
        'pool': 'max',
        "layers_n_channels": [16, 32, 64, 128],
        'layers_n_non_lins': 2,
    }


    # TF
    start_tf = time.time()

    import tensorflow as tf
    from unet_tf import get_zerofilled as GZeroF_TF
    from unet_tf import filename_to_image_and_kspace as FTIAK_TF
    from autre.fastmri_tf.models.functional_models.unet import unet

    image , kspace = FTIAK_TF(filename)
    target = tf.abs(image)

    zero_filled = GZeroF_TF(kspace, af=accelerations)

    model = unet(input_size=zero_filled.shape, lr=lr, **run_params)
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss_list = []
    zero_filled = tf.expand_dims(zero_filled, 0)
    for epoch in range(num_epochs):
        
        with tf.GradientTape() as tape:
            outputs = model(zero_filled)
            loss = criterion(outputs, target)

        loss_list.append(loss)

        gradients = tape.gradient(loss_list, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f'Epoch [{epoch + 1}/{num_epochs}]')


    end_tf = time.time()


    # PyTorch
    start_torch = time.time()

    import torch
    from unet_torch import get_zerofilled as GZeroF_Torch
    from unet_torch import filename_to_image_and_kspace as FTIAK_Torch
    from fastmri.models.unet import Unet

    image_torch, kspace_torch = FTIAK_Torch(filename)
    target_torch = torch.abs(image_torch)

    zero_filled_torch = GZeroF_Torch(kspace_torch)

    model_torch = Unet(in_chans=1, out_chans=1, chans=chans, num_pool_layers=num_pool_layers)
    criterion_torch = torch.nn.L1Loss()
    optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=lr)

    loss_list_torch = []
    for epoch in range(num_epochs):

        optimizer_torch.zero_grad()
        outputs_torch = model_torch(zero_filled_torch)
        loss_torch = criterion_torch(outputs_torch, target_torch)
        loss_list_torch.append(loss_torch.item())

        loss_torch.backward()
        optimizer_torch.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]')

    end_torch = time.time()


    # # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    # np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)
    # print("Values are close.")

    print("tf time :", end_tf - start_tf)
    print("torch time : ", end_torch - start_torch)

    outputs = tf.squeeze(outputs, axis= [0,len(outputs.shape)-1])
    outputs_torch = torch.squeeze(outputs_torch, dim = [ 0, len(outputs_torch.shape)-1])

    print("tf shape :", outputs.shape)
    print("torch shape : ", outputs_torch.shape)

    # # Assurez-vous que les formes sont correctes
    # assert tf_output.shape == pt_output.shape
    # print("Shapes match.")





file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
test_combine_images(file_path)

