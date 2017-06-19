import numpy as np
import cv2


def get_dark_channel(img, patch_size):
    """Get the DCP from RGB image.

    Parameters
    -----------
    img: an M*N*3 numpy array image, M is the heigh, N is the width,
       3 are the channels of image

    Return
    -----------
    An M*N array for the Dark Channel Prior.

    """

    M, N, _ = img.shape
    darkch = np.zeros((M, N), dtype=np.float32)
    padded = np.pad(img, ((patch_size // 2, patch_size // 2),
                          (patch_size // 2, patch_size // 2), (0, 0)), 'edge')
    cv2.imshow('pad', padded)
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(
            padded[i:i + patch_size, j:j + patch_size, :])  # CVPR09, eq.5
    cv2.imshow('darkch', darkch)
    return darkch


def get_atmosphere(img, darkch, p):
    """Get the atmosphere light atmos. (4.3)

    Parameters
    -----------
    img: an M*N*3 numpy array image, M is the heigh, N is the width,
       3 are the channels of image.
    darkch: Dark Channel Prior, M*N arrar.
    p: The top p portion to take.

    Return
    ----------
    atmos 1x3 element array, which is the atmos for each channel.

    """
    M, N = darkch.shape
    flatimg = img.reshape(M * N, 3)
    flatdark = darkch.ravel()
    # find top M * N * p indexes
    searchidx = (-flatdark).argsort()[:int(M * N * p)]

    # return the highest intensity for each channel
    return np.max(flatimg.take(searchidx, axis=0), axis=0)


def get_transmission(img, atmos, darkch, omega, patch_size):
    """Get the transmission esitmate in the (RGB) image data.
    Parameters
    -----------
    img:          the M * N * 3 RGB image data ([0, L-1]) as numpy array
    atmos:          a 3-element array containing atmosphere light
                ([0, L-1]) for each channel
    darkch:     the dark channel prior of the image as an M * N numpy array
    omega:      bias for the estimate
    patch_size: window size for the estimate
    Return
    -----------
    an M * N array containing the transmission rate ([0.0, 1.0])
    """
    raw_t = 1 - omega * \
        get_dark_channel(img / atmos, patch_size)  # CVPR09, eq.12

    cv2.imshow('raw_t', raw_t)
    # refinement by guided filter
    refine_t = cv2.ximgproc.guidedFilter(img, raw_t, 50, 1e-4)
    return refine_t


def get_radiance(img, atmos, t, t0):
    """Recover the radiance from raw image data with atmosphere light
       and transmission rate estimate.

        Parameters
    ----------
    img:      M * N * 3 data as numpy array for the hazy image
    atmos:      a 3-element array containing atmosphere light
            ([0, L-1]) for each channel
    t:      estimate for the transmission rate
    t0:     The constant, which can make the image good better.
    Return
    ----------
    M * N * 3 numpy array for the recovered radiance
    """
    t_clip = np.clip(t, a_min=t0, a_max=1.0)
    t_tiled = np.zeros_like(img, dtype=np.float32)  # tiled to M * N * 3
    for i in range(3):
        t_tiled[:, :, i] = t_clip
    cv2.imshow('refine_t', t_clip)
    return (img - atmos) / t_tiled + atmos


def brightness_equalize(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 2] = cv2.equalizeHist(img_hsv[..., 2])
    img_equal = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv equalize', img_equal)

    # Lab colorspace has alittle bit brighter, not knowing why
    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # img_lab[..., 0] = cv2.equalizeHist(img_lab[..., 0])
    # img_equal = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # cv2.imshow('lab equalize', img_equal)
    cv2.waitKey(0)
    return img_equal


def dehaze(img, patch_size, top_p, t0, omega):
    darkch = get_dark_channel(img, patch_size)
    atmos = get_atmosphere(img, darkch, top_p)
    refine_t = get_transmission(img, atmos, darkch, omega, patch_size)
    img_dehaze = get_radiance(img, atmos, refine_t, t0)
    # img_dehaze = img_dehaze.astype(np.uint8)
    img_dehaze = (img_dehaze / np.max(img_dehaze) * 255).astype(np.uint8)
    cv2.imshow('dehaze img(wo equalize)', img_dehaze)
    img_equal = brightness_equalize(img_dehaze)
    return img_equal
