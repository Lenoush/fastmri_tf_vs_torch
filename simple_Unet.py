"""A simple U-Net model for MRI image reconstruction using the FastMRI dataset."""

import torch
import glob
import h5py
import matplotlib.pyplot as plt

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models.unet import Unet


class Unet_Model(torch.nn.Module):
    """
    A class for training a U-Net model for MRI image reconstruction.

    Attributes
    ----------
    path : str
        Path to the dataset.
    name_for_save : str
        Name of the file to save the trained model.
    num_epochs : int
        Number of epochs for training.
    num_pool_layers : int
        Number of pooling layers in the U-Net model.
    lr : float
        Learning rate for the optimizer.
    multi_coil : bool
        Indicates if the multi-coil data is used for reconstruction.
    """

    def __init__(
        self,
        path: str,
        name_for_save: str,
        num_epochs: int = 200,
        num_pool_layers: int = 4,
        lr: float = 1e-3,
        multi_coil: bool = True,
    ):
        """
        Initialize the U-Net model, loss function, and optimizer.

        Parameters
        ----------
        path : str
            Path to the dataset.
        name_for_save : str
            Name of the file to save the trained model.
        num_epochs : int, optional
            Number of epochs for training (default is 200).
        num_pool_layers : int, optional
            Number of pooling layers in the U-Net model (default is 4).
        lr : float, optional
            Learning rate for the optimizer (default is 1e-3).
        multi_coil : bool, optional
            Indicates if the multi-coil data is used for reconstruction.
            Default is True.
        """
        super().__init__()
        self.path = path
        self.num_epochs = num_epochs
        self.num_pool_layers = num_pool_layers
        self.lr = lr
        self.name_for_save = name_for_save
        self.multi_coil = multi_coil

        self.model = Unet(in_chans=1, out_chans=1, num_pool_layers=num_pool_layers)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_list = []
        self.outputs10 = []

    def filename_to_image_and_kspace(self, one_path):
        """
        Load the k-space data from the given file and computes the corresponding image.

        Parameters
        ----------
        one_path : str
            Path to the file containing the data.

        Returns
        -------
        tuple
            A tuple containing the image and the corresponding k-space data.
        """
        kspace = self.load_and_transform(one_path)
        image = torch.fft.fftshift(torch.fft.ifft2(kspace))
        if self.multi_coil:
            image = self.virtual_coil_reconstruction(image)
            image = image.unsqueeze(1)
            kspace = torch.fft.fft2(image)
        return image, kspace

    def load_and_transform(self, file_name):
        """
        Load the k-space data from an h5 file and converts it to a tensor.

        Parameters
        ----------
        file_name : str
            Path to the h5 file containing the k-space data.

        Returns
        -------
        torch.Tensor
            The k-space data as a complex tensor.
        """
        hf = h5py.File(file_name, "r")
        kspace = hf["kspace"][()]
        kspace = torch.tensor(kspace, dtype=torch.complex64)
        return kspace

    def ortho_fft2d(self, image):
        """
        Perform an orthogonal 2D FFT on the given image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The k-space data after applying FFT.
        """
        image = image.to(dtype=torch.complex64)
        scaling_norm = torch.sqrt(
            torch.tensor(image.size(-2) * image.size(-1), dtype=torch.float32)
        ).to(image.dtype)
        if len(image.shape) == 4:
            # multicoil case
            ncoils = image.shape[1]
        n_slices = image.shape[0]
        i_shape_x = image.shape[-2]
        i_shape_y = image.shape[-1]
        shifted_image = torch.fft.fftshift(image)
        batched_shifted_image = shifted_image.view(-1, i_shape_x, i_shape_y)
        batched_shifted_kspace = torch.stack(
            [torch.fft.fft2(img) for img in batched_shifted_image]
        )
        if len(image.shape) == 4:
            # multicoil case
            kspace_shape = [n_slices, ncoils, i_shape_x, i_shape_y]
        elif len(image.shape) == 3:
            kspace_shape = [n_slices, i_shape_x, i_shape_y]
        else:
            kspace_shape = [i_shape_x, i_shape_y]
        shifted_kspace = batched_shifted_kspace.view(kspace_shape)
        kspace = torch.fft.ifftshift(shifted_kspace)
        return kspace / scaling_norm

    def ortho_ifft2d(self, kspace):
        """
        Perform an orthogonal 2D inverse FFT on the given k-space data.

        Parameters
        ----------
        kspace : torch.Tensor
            The input k-space tensor.

        Returns
        -------
        torch.Tensor
            The image data after applying inverse FFT.
        """
        scaling_norm = torch.sqrt(
            torch.tensor(kspace.size(-2) * kspace.size(-1), dtype=torch.float32)
        ).to(kspace.dtype)
        if len(kspace.shape) == 4:
            # multicoil case
            ncoils = kspace.shape[1]
        n_slices = kspace.shape[0]
        k_shape_x = kspace.shape[-2]
        k_shape_y = kspace.shape[-1]
        shifted_kspace = torch.fft.ifftshift(kspace)
        batched_shifted_kspace = shifted_kspace.view((-1, k_shape_x, k_shape_y))
        batched_shifted_image = torch.stack(
            [torch.fft.ifft2(ksp) for ksp in batched_shifted_kspace]
        )
        if len(kspace.shape) == 4:
            # multicoil case
            image_shape = [n_slices, ncoils, k_shape_x, k_shape_y]
        elif len(kspace.shape) == 3:
            image_shape = [n_slices, k_shape_x, k_shape_y]
        else:
            image_shape = [k_shape_x, k_shape_y]
        shifted_image = batched_shifted_image.view(image_shape)
        image = torch.fft.fftshift(shifted_image)
        return scaling_norm * image

    def virtual_coil_reconstruction(self, imgs):
        """
        Reconstruct an image from multi-coil images to single-coil images.

        Parameters
        ----------
        imgs : torch.Tensor
            The multi-coil images.

        Returns
        -------
        torch.Tensor
            The reconstructed single-coil image.
        """
        img_sh = imgs.shape
        dimension = len(img_sh) - 2
        # Compute first the virtual coil
        weights = torch.sum(torch.abs(imgs), dim=1) + 1e-16
        phase_reference = (
            torch.angle(torch.sum(imgs, dim=tuple(2 + torch.arange(len(img_sh) - 2))))
            .clone()
            .detach()
        )
        expand = [Ellipsis, *((None,) * (len(img_sh) - 2))]

        reference = (
            imgs
            / weights[:, None, ...].to(torch.complex64)
            / torch.exp(1j * phase_reference)[expand]
        )
        virtual_coil = torch.sum(reference, dim=1)

        difference_original_vs_virtual = torch.conj(imgs) * virtual_coil.unsqueeze(1)
        hanning = torch.hann_window(img_sh[-dimension])
        for d in range(dimension - 1):
            hanning = hanning.unsqueeze(-1) * torch.hann_window(img_sh[dimension + d])
        hanning = hanning.to(torch.complex64)

        if dimension == 3:
            fft_result = torch.fft.fftn(difference_original_vs_virtual)
            hanning = torch.fft.fftshift(hanning)
            difference_original_vs_virtual = torch.fft.ifftn(fft_result * hanning)
        else:
            fft_result = self.ortho_fft2d(difference_original_vs_virtual)
            shape_want = fft_result.shape[-1]
            hanning = hanning[:, :shape_want]
            difference_original_vs_virtual = self.ortho_ifft2d(fft_result * hanning)

        img_comb = torch.sum(
            imgs
            * torch.exp(
                1j * torch.angle(difference_original_vs_virtual.to(torch.complex64))
            ),
            dim=1,
        )

        return img_comb

    def get_zerofilled(self, kspace, mask_type="random"):
        """
        Generate a zero-filled image from masked k-space data.

        Parameters
        ----------
        kspace : torch.Tensor
            The k-space data.
        mask_type : str, optional
            Type of mask to apply on the k-space.
            Default to 'random'.

        Returns
        -------
        torch.Tensor
            The zero-filled image.
        """
        center_fractions = [0.08]
        accelerations = [4]
        mask_func = create_mask_for_mask_type(
            mask_type, center_fractions, accelerations
        )

        if len(kspace.shape) == 4:
            kspace = kspace.unsqueeze(-1)

        zero_filled_list = []
        for batch in range(kspace.shape[0]):
            mask, _ = mask_func(kspace.shape)
            masked_kspace = kspace[batch] * mask
            zero_filled = torch.fft.ifftn(masked_kspace)
            zero_filled_list.append(zero_filled)

        zero_filled = torch.cat(zero_filled_list)
        zero_filled = zero_filled.squeeze(-1)

        return zero_filled

    def train_model(self):
        """
        Trains the U-Net model.

        Returns
        -------
        list
            A list of model outputs and loss values after every 10th epochs.
        """
        files_paths = glob.glob(self.path + "*.h5")
        if not files_paths:
            raise ValueError(f"No h5 files found at path {self.path}")

        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            for i, file_path in enumerate(files_paths):
                print(f"File [{i+1}/{len(files_paths)}]: {file_path}")

                image, kspace = self.filename_to_image_and_kspace(file_path)
                target = image.abs()
                zero_filled = self.get_zerofilled(kspace)
                zero_filled = torch.abs(zero_filled)

                self.optimizer.zero_grad()
                outputs = self.model(zero_filled)
                loss = self.criterion(outputs, target)
                self.loss_list.append(loss.item())
                if epoch % 10 == 0:
                    self.outputs10.append([outputs, loss.item()])
                loss.backward()
                self.optimizer.step()

        torch.save(self.model.state_dict(), self.name_for_save)
        return self.outputs10


if __name__ == "__main__":
    model = Unet_Model(
        path="/volatile/FastMRI/brain_multicoil_train/multicoil_train/dataset/",
        # path_jz= "/gpfsscratch/rech/hih/commun/fastmri2024/multicoil_train/",
        num_epochs=1,
        name_for_save="./model.pth",
    )
    outputs = model.train_model()
    # # outputs = list of list with 2 elements : [output, loss]
    print(outputs[0][0].shape)  # shape of the first output
    print(outputs[0][1])  # loss of the first output

    # # Display the first output
    plt.imshow(outputs[0][0][0, 0, :, :].abs().detach().numpy())
    plt.show()
