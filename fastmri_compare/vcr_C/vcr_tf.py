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
    img_comb = tf.signal.fftshift(img_comb)
    return img_comb


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