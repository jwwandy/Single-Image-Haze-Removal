import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.exposure
import skimage
import skimage.io
import os
from cv2.ximgproc import guidedFilter


def show_img(title, img, cmap=None):
    plt.figure()
    plt.title(title)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)


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
    show_img('pad', padded)
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(
            padded[i:i + patch_size, j:j + patch_size, :])  # CVPR09, eq.5
    show_img('darkch', darkch, cmap='gray')
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


def get_transmission(img, atmos, omega, patch_size):
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

    show_img('raw_t', raw_t, cmap='gray')
    # refinement by guided filter
    refine_t = guidedFilter(img.astype(np.float32),
                            raw_t.astype(np.float32), 50, 1e-4)
    return raw_t, refine_t


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
    show_img('refine_t', t_clip, cmap='gray')
    radiance = np.clip((img - atmos) / t_tiled + atmos, a_min=0, a_max=255)
    return radiance


def brightness_equalize(img):
    img_hsv = skimage.color.rgb2hsv(img)
    img_hsv[..., 2] = skimage.exposure.equalize_hist(img_hsv[..., 2])
    img_equal = skimage.color.hsv2rgb(img_hsv)

    show_img('hsv equalize', img_equal)

    # Lab colorspace has alittle bit brighter, not knowing why
    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # img_lab[..., 0] = cv2.equalizeHist(img_lab[..., 0])
    # img_equal = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # show_img('lab equalize', img_equal)
    return img_equal


def dehaze(img, patch_size, top_p, t0, omega, outdir):
    darkch = get_dark_channel(img, patch_size)
    skimage.io.imsave(os.path.join(outdir, 'dark.jpg'), np.uint8(darkch))
    atmos = get_atmosphere(img, darkch, top_p)
    raw_t, refine_t = get_transmission(img, atmos, omega, patch_size)

    skimage.io.imsave(os.path.join(
        outdir, 'raw_transmission.jpg'), (255 * raw_t).astype(np.uint8))
    skimage.io.imsave(os.path.join(
        outdir, 'refine_transmission.jpg'), (255 * refine_t).astype(np.uint8))
    img_dehaze = get_radiance(img, atmos, refine_t, t0)
    skimage.io.imsave(os.path.join(outdir, 'noequalize.jpg'),
                      np.uint8(img_dehaze))
    show_img('dehaze img(wo equalize)', img_dehaze.astype(np.uint8))
    img_equal = brightness_equalize((img_dehaze.astype(np.uint8)))
    plt.show(block=False)
    return img_equal
