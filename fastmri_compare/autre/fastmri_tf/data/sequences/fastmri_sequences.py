"""keras Sequences used for fastMRI data"""
import glob
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from ..utils.h5 import from_file_to_kspace, from_test_file_to_mask_and_kspace, from_train_file_to_image_and_kspace, from_file_to_contrast
from ..utils.h5 import from_multicoil_train_file_to_image_and_kspace
from ...evaluate.reconstruction.zero_filled_reconstruction import zero_filled_cropped_recon, zero_filled_recon
from ..utils.masking.gen_mask import gen_mask
from ..utils.normalize import normalize, normalize_instance


class fastMRI2DSequence(Sequence):
    """The base abstract class for using the fastMRI data in keras.
    You need to specify the path to the type of data you want, the mode of
    the sequence and its acceleration rate.
    This will by default enumerate volumes.

    Parameters:
    path (str): the path to the data of this sequence. The data must be in
    h5 files.
    mode (str): the mode of sequence in ['training', 'validation', 'testing'].
    The mode training is to be used for both validation and training data,
    when training the neural network. The validation mode is to be used when
    evaluating the neural network offline, with potentially other
    reconstruction steps used afterwards. The testing mode is when handling
    test data.
    af (int): the acceleration factor.

    Raises:
    ValueError: when no h5 files can be found in the path directory.
    """
    train_modes = ('training', 'validation')

    def __init__(self, path, mode='training', af=4, contrast=None):
        self.path = path
        self.mode = mode
        self.af = af
        self.contrast = contrast

        self.filenames = glob.glob(path + '*.h5')
        if self.contrast is not None:
            contrast_filenames = list()
            for filename in self.filenames:
                contrast = from_file_to_contrast(filename)
                if contrast == self.contrast:
                    contrast_filenames.append(filename)
            self.filenames = contrast_filenames
        if not self.filenames:
            raise ValueError(
                'No h5 files of given contrast {} at path {}'.format(
                    contrast, path
                )
            )
        self.filenames.sort()

        if mode == 'testing':
            af_filenames = list()
            for filename in self.filenames:
                mask, _ = from_test_file_to_mask_and_kspace(filename)
                mask_af = len(mask) / sum(mask)
                if af == 4 and mask_af < 5.5 or af == 8 and mask_af > 5.5:
                    af_filenames.append(filename)
            self.filenames = af_filenames


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if self.mode in type(self).train_modes:
            return self.get_item_train(filename)
        else:
            return self.get_item_test(filename)


    def get_item_train(self, filename):
        pass

    def get_item_test(self, filename):
        pass


class SingleSliceSequence(fastMRI2DSequence):
    """
    The particularity of this sequence is that it enumerates slices instead
    of volumes, therefore allowing for smaller batch sizes while still visiting
    the whole dataset.
    """
    train_modes = ('training', 'validation')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_filename_and_position = list()
        for filename in self.filenames:
            kspaces = from_file_to_kspace(filename)
            filename_and_position = [(filename, i) for i in range(kspaces.shape[0])]
            self.idx_to_filename_and_position += filename_and_position

    def __len__(self):
        return len(self.idx_to_filename_and_position)

    def __getitem__(self, idx):
        if self.mode in type(self).train_modes:
            return self.get_item_train(idx)
        else:
            return self.get_item_test(idx)


class Untouched2DSequence(fastMRI2DSequence):
    def get_item_train(self, filename):
        """Get the images and the kspaces of the volume at filename.

        Parameters:
        filename (str): the name of the h5 file containing the images and
        the kspaces

        Returns:
        tuple (ndarray, ndarray): the images and the kspaces corresponding to
        the volume in HWC format (i.e. with an extra dimension).
        """
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        images = images[..., None]
        kspaces = kspaces[..., None]
        return images, kspaces

    def get_item_test(self, filename):
        """Get the kspaces and mask of the volume at filename.

        Parameters:
        filename (str): the name of the h5 file containing the kspaces and the
        mask

        Returns:
        tuple (list, ndarray): the mask and the kspaces corresponding to
        the volume in HWC format (i.e. with an extra dimension).
        """
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        kspaces = kspaces[..., None]
        return mask, kspaces


class Masked2DSequence(Untouched2DSequence):
    """This sequence allows to generate a mask on-the-fly when enumerating
    training or validation examples. It also allows you to restrict the
    training to only innermost parts of the volumes, and select randomly
    a slice when training. Finally, you can scale the values of the
    kspaces and images by a factor.

    Parameters:
    inner_slices (int): the number of inner slices you want to consider when
    enumerating the volumes.
    rand (bool): whether you want to only pick one random slice from the
    considered slices when enumerating the volumes.
    scale_factor (float): the factor by which to multiply the kspaces and the
    images, if scaling is needed
    """
    def __init__(self, *args, inner_slices=None, rand=False, scale_factor=1, mask_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand
        self.scale_factor = scale_factor
        self.mask_seed = mask_seed

    def get_item_train(self, filename):
        """Get a training triplet from the file at filename.

        This method will get the kspaces and images at filename, create a mask
        on-the-fly, mask the kspaces with it, select only the relevant slices,
        and return a tuple ((kspaces, mask), images).

        Parameters:
        filename (str): the name of the h5 file containing the images and
        the kspaces.

        Returns:
        tuple ((ndarray, ndarray), ndarray): the masked kspaces, mask and images
        corresponding to the volume in NHWC format (mask is NHW).
        """
        images, kspaces = super(Masked2DSequence, self).get_item_train(filename)
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af, seed=self.mask_seed)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
        kspaces *= mask_batch
        mask_batch = mask_batch[..., 0]
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            if self.rand:
                i_slice = random.randint(slice_start, slice_start + self.inner_slices - 1)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
            images = images[selected_slices]
            mask_batch = mask_batch[selected_slices]
        scale_factor = self.scale_factor
        kspaces_scaled = kspaces * scale_factor
        images_scaled = images * scale_factor
        return ([kspaces_scaled, mask_batch], images_scaled)

    def get_item_test(self, filename):
        """Get the kspaces and mask of the volume at filename.

        Parameters:
        filename (str): the name of the h5 file containing the kspaces and the
        mask

        Returns:
        tuple (ndarray, ndarray): the mask and the kspaces corresponding to
        the volume in NHWC format (mask is NHW).
        """
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        k_shape = kspaces[0].shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)
        kspaces_scaled = kspaces * self.scale_factor
        return kspaces_scaled, mask_batch


class ZeroFilled2DSequence(fastMRI2DSequence):
    """
    This sequence generates pre-reconstructed examples, with zero filling
    and cropping. You can potentially have the examples normalized.

    Parameters:
    norm (bool): whether you want to normalize each volume. When using `norm`
    in 'validation' mode, the `get_item_train` method will return the mean
    and standard deviation used for normalization as well.
    """
    def __init__(self, *args, norm=False, mask_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.mask_seed = mask_seed


    def get_item_train(self, filename):
        """Get the reconstructed images and the images of the volume.

        This method will generate a mask on-the-fly, mask the kspaces and then
        do a zero-filled reconstruction. If `norm` is True, the images
        (both reconstructed and ground truth) are nomalized using the mean
        and the standard deviation computed with the reconstructed image.

        Parameters:
        filename (str): the name of the h5 file containing the images and
        the kspaces

        Returns:
        tuple (ndarray, ndarray) or tuple (ndarray, ndarray, float, float):
        the reconstructed masked kspaces and the images corresponding to the
        volume in NHWC format. If `norm` is True, and `mode` is 'validation',
        mean and stddev are also returned.
        """
        po, kspaces = from_multicoil_train_file_to_image_and_kspace(filename)
        # images, kspaces = from_train_file_to_image_and_kspace(filename)
        images = tf.signal.fftshift(tf.signal.ifft2d(kspaces))
        images = virtual_coil_reconstruction(images)
        kspaces = tf.signal.fft2d(images)

        mask = gen_mask(kspaces[0], accel_factor=self.af, seed=self.mask_seed)
        fourier_mask = np.repeat(mask.astype(np.float64), kspaces[0].shape[0], axis=0)
        img_batch = list()
        zero_img_batch = list()
        if self.norm and self.mode == 'validation':
            means = list()
            stddevs = list()

        for kspace, image in zip(kspaces, images):

            zero_filled_rec = zero_filled_cropped_recon(kspace * fourier_mask)

            # if self.norm:
                # zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                # image = normalize(image, mean, std, eps=1e-11)
                # if self.mode == 'validation':
                #     means.append(mean)
                #     stddevs.append(std)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
            image = image[..., None]
            img_batch.append(image)

        zero_img_batch = np.array(zero_img_batch)
        img_batch = np.array(img_batch)
        if self.norm and self.mode == 'validation':
            return zero_img_batch, img_batch, means, stddevs
        else:
            return (zero_img_batch, img_batch)


    def get_item_test(self, filename):
        """Get the reconstructed images of the volume.

        This method will do a zero-filled reconstruction.
        If `norm` is True, the images are nomalized using the mean
        and the standard deviation.

        Parameters:
        filename (str): the name of the h5 file containing the kspaces and the
        mask.

        Returns:
        ndarray or tuple (ndarray, float, float): the reconstructed masked kspaces and
        the images corresponding to the volume in NHWC format. If `norm`
        is True, and `mode` is 'validation', mean and stddev are also returned.
        """
        _, kspaces = from_test_file_to_mask_and_kspace(filename)
        zero_img_batch = list()
        means = list()
        stddevs = list()
        for kspace in kspaces:
            zero_filled_rec = zero_filled_cropped_recon(kspace)
            if self.norm:
                zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                means.append(mean)
                stddevs.append(std)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
        zero_img_batch = np.array(zero_img_batch)
        if self.norm:
            return zero_img_batch, means, stddevs
        else:
            return zero_img_batch


class KIKISequence(Untouched2DSequence):
    """This sequence allows to generate a mask on-the-fly when enumerating
    training or validation examples. It also allows you to restrict the
    training to only innermost parts of the volumes, and select randomly
    a slice when training. Finally, you can scale the values of the
    kspaces and images by a factor.
    The target values are not cropped or in magnitude, but the actual ones.

    Parameters:
    inner_slices (int): the number of inner slices you want to consider when
    enumerating the volumes.
    rand (bool): whether you want to only pick one random slice from the
    considered slices when enumerating the volumes.
    scale_factor (float): the factor by which to multiply the kspaces and the
    images, if scaling is needed
    space (str): the space of the sequence, i.e. whether the target value is
    the ground truth k-space (K) or the ground-truth image (I).
    """
    def __init__(self, *args, inner_slices=None, rand=False, scale_factor=1, space='K', **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand
        self.scale_factor = scale_factor
        self.space = space

    def get_item_train(self, filename):
        """Get a training triplet from the file at filename.

        This method will get the kspaces and images at filename, create a mask
        on-the-fly, mask the kspaces with it, select only the relevant slices,
        and return a tuple ((kspaces, mask), images).

        Parameters:
        filename (str): the name of the h5 file containing the images and
        the kspaces.

        Returns:
        tuple ((ndarray, ndarray), ndarray): the masked kspaces, mask and images
        corresponding to the volume in NHWC format (mask is NHW).
        """
        _, kspaces = super(KIKISequence, self).get_item_train(filename)
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
        kspaces_masked = kspaces * mask_batch
        mask_batch = mask_batch[..., 0]
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            if self.rand:
                i_slice = random.randint(slice_start, slice_start + self.inner_slices)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
            kspaces_masked = kspaces_masked[selected_slices]
            mask_batch = mask_batch[selected_slices]
        scale_factor = self.scale_factor
        kspaces_masked_scaled = kspaces_masked * scale_factor
        kspaces_scaled = kspaces * scale_factor
        if self.space == 'K':
            return ([kspaces_masked_scaled, mask_batch], kspaces_scaled)
        elif self.space == 'I':
            images = zero_filled_recon(kspaces_scaled[..., 0])[..., None]
            return ([kspaces_masked_scaled, mask_batch], images)






import numpy as np
import tensorflow as tf
import multiprocessing


def virtual_coil_reconstruction(imgs):
    """
    Calculate the combination of all coils using virtual coil reconstruction

    Parameters
    ----------
    imgs: np.ndarray
        The images reconstructed channel by channel
        in shape [batch_size, Nch, Nx, Ny, Nz]

    Returns
    -------
    img_comb: np.ndarray
        The combination of all the channels in a complex valued
        in shape [batch_size, Nx, Ny]
    """
    img_sh = imgs.shape
    dimension = len(img_sh)-2
    # Compute first the virtual coil
    weights = tf.math.reduce_sum(tf.abs(imgs), axis=1) + 1e-16
    phase_reference = tf.cast(
        tf.math.angle(tf.math.reduce_sum(
            imgs,
            axis=(2+np.arange(len(img_sh)-2))
        )),
        tf.complex64
    )
    expand = [..., *((None, ) * (len(img_sh)-2))]
    reference = imgs / tf.cast(weights[:, None, ...], tf.complex64) / \
        tf.math.exp(1j * phase_reference)[expand]
    virtual_coil = tf.math.reduce_sum(reference, axis=1)
    imgs = tf.cast(imgs, tf.complex64)## ajout LENA 
    difference_original_vs_virtual = tf.math.conj(imgs) * virtual_coil[:, None]
    # Hanning filtering in readout and phase direction
    hanning = tf.signal.hann_window(img_sh[-dimension])
    for d in range(dimension-1):
        hanning = tf.expand_dims(hanning, axis=-1) * tf.signal.hann_window(img_sh[dimension + d])
    hanning = tf.cast(hanning, tf.complex64)
    # Removing the background noise via low pass filtering
    if dimension == 3:    
        difference_original_vs_virtual = tf.signal.ifft3d(
            tf.signal.fft3d(difference_original_vs_virtual) * tf.signal.fftshift(hanning)
        )
    else:
        fft_result = ortho_fft2d(difference_original_vs_virtual) 
        shape_want = fft_result.shape[-1]
        hanning = hanning[:, :shape_want]
        difference_original_vs_virtual = ortho_ifft2d(fft_result * hanning)
    
    img_comb = tf.math.reduce_sum(
        imgs *
        tf.math.exp(
            1j * tf.cast(tf.math.angle(difference_original_vs_virtual), tf.complex64)),
        axis=1
    )
    return img_comb



def ortho_ifft2d(kspace):
    axes = [len(kspace.shape) - 2, len(kspace.shape) - 1]
    scaling_norm = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')), kspace.dtype)
    if len(kspace.shape) == 4:
        # multicoil case
        ncoils = tf.shape(kspace)[1]
    n_slices = tf.shape(kspace)[0]
    k_shape_x = tf.shape(kspace)[-2]
    k_shape_y = tf.shape(kspace)[-1]
    shifted_kspace = tf.signal.ifftshift(kspace, axes=axes)
    batched_shifted_kspace = tf.reshape(shifted_kspace, (-1, k_shape_x, k_shape_y))
    batched_shifted_image = tf.map_fn(
        tf.signal.ifft2d,
        batched_shifted_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    if len(kspace.shape) == 4:
        # multicoil case
        image_shape = [n_slices, ncoils, k_shape_x, k_shape_y]
    elif len(kspace.shape) == 3:
        image_shape = [n_slices, k_shape_x, k_shape_y]
    else:
        image_shape = [k_shape_x, k_shape_y]
    shifted_image = tf.reshape(batched_shifted_image, image_shape)
    image = tf.signal.fftshift(shifted_image, axes=axes)
    return scaling_norm * image


def ortho_fft2d(image):
    image = tf.cast(image, 'complex64')
    axes = [len(image.shape) - 2, len(image.shape) - 1]
    scaling_norm = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), 'float32')), image.dtype)
    if len(image.shape) == 4:
        # multicoil case
        ncoils = tf.shape(image)[1]
    n_slices = tf.shape(image)[0]
    i_shape_x = tf.shape(image)[-2]
    i_shape_y = tf.shape(image)[-1]
    shifted_image = tf.signal.fftshift(image, axes=axes)
    batched_shifted_image = tf.reshape(shifted_image, (-1, i_shape_x, i_shape_y))
    batched_shifted_kspace = tf.map_fn(
        tf.signal.fft2d,
        batched_shifted_image,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    if len(image.shape) == 4:
        # multicoil case
        kspace_shape = [n_slices, ncoils, i_shape_x, i_shape_y]
    elif len(image.shape) == 3:
        kspace_shape = [n_slices, i_shape_x, i_shape_y]
    else:
        kspace_shape = [i_shape_x, i_shape_y]
    shifted_kspace = tf.reshape(batched_shifted_kspace, kspace_shape)
    kspace = tf.signal.ifftshift(shifted_kspace, axes=axes)
    return kspace / scaling_norm