import numpy as np
import cv2


R, G, B = 0, 1, 2
L = 256


def get_dark_channel(I, w):
    """Get the DCP from RGB image.

    Parameters
    -----------
    I: an M*N*3 numpy array image, M is the heigh, N is the width,
       3 are the channels of image
    w: window size

    Return
    -----------
    An M*N array for the Dark Channel Prior.

    """

    M, N, _ = I.shape
    darkch = np.zeros((M, N))
    padded = np.pad(I, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light A. (4.3)

    Parameters
    -----------
    I: an M*N*3 numpy array image, M is the heigh, N is the width,
       3 are the channels of image.
    darkch: Dark Channel Prior, M*N arrar.
    p: The top p portion to take.

    Return
    ----------
    A 1x3 element array, which is the A for each channel.

    """
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    # find top M * N * p indexes
    searchidx = (-flatdark).argsort()[:int(M * N * p)]
    # print('atmosphere light region:', [(i / N, i % N) for i in searchidx])

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def get_transmission(I, A, darkch, omega, w):
    """Get the transmission esitmate in the (RGB) image data.
    Parameters
    -----------
    I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:       a 3-element array containing atmosphere light
             ([0, L-1]) for each channel
    darkch:  the dark channel prior of the image as an M * N numpy array
    omega:   bias for the estimate
    w:       window size for the estimate
    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    raw_t = 1 - omega * get_dark_channel(I / A, w)  # CVPR09, eq.12
    cv2.imshow('raw_t', raw_t)
    # refinement by guided filter
    refine_t = cv2.ximgproc.guidedFilter(
        I.astype(np.float32), raw_t.astype(np.float32), 20, 1e-3)
    return refine_t


def get_radiance(I, A, t, t0):
    """Recover the radiance from raw image data with atmosphere light
       and transmission rate estimate.

        Parameters
    ----------
    I:      M * N * 3 data as numpy array for the hazy image
    A:      a 3-element array containing atmosphere light
            ([0, L-1]) for each channel
    t:      estimate for the transmission rate
    t0:     The constant, which can make the image good better.
    Return
    ----------
    M * N * 3 numpy array for the recovered radiance
    """
    # print('A:', A.dtype)
    t_clip = np.clip(t, a_min=t0, a_max=1.0)
    t_tiled = np.zeros_like(I, dtype=np.float32)  # tiled to M * N * 3
    for i in range(3):
        t_tiled[:, :, i] = t_clip
    cv2.imshow('refine_t', t_clip)
    return (I.astype(np.float32) - A.astype(np.float32)) / t_tiled.astype(np.float32) + A.astype(np.float32)


def dehaze(I, patch_size, top_p, t0, omega):
    darkch = get_dark_channel(I, patch_size)
    A = get_atmosphere(I, darkch, top_p)
    print(A)
    refine_t = get_transmission(I, A, darkch, omega, patch_size)
    img_rehaze = get_radiance(I, A, refine_t, t0)
    cv2.imshow('img', img_rehaze.astype(np.uint8))
    cv2.waitKey(0)
    return img_rehaze
