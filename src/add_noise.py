import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import argparse
from skimage.morphology import opening
from skimage.morphology import closing


parser = argparse.ArgumentParser(
    description="Add the noise to the input image.")
parser.add_argument("-i", "--impath", type=str, required=True,
                    help="input image path")
parser.add_argument("--t", type=str, default='gauss',
                    help="The type of noise.")
parser.add_argument("-o", "--out", type=str, default="../result/out_noisy.jpg",
                    help="output image path (default: ../result/out_noisy.jpg)")


def show_img(title, img, cmap=None):
    plt.figure()
    plt.title(title)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)


def noisy(image, noise_typ):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 500
        var = 50
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = np.abs(gauss.reshape(row, col, ch))
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    '''
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    '''

if __name__ == '__main__':
    noise_type = ['gauss', 's&p']
    args = parser.parse_args()
    if args.t not in noise_type:
        print("There are only {} in noise type.".format(noise_type))
    img = (skimage.io.imread(args.impath).astype(np.float32))
    img_noisy = noisy(img, args.t)
    img_denoisy = img_noisy
    for i in range(100):
        img_denoisy = closing(opening(img_denoisy))

    skimage.io.imsave(args.out, img_denoisy / np.max(img_denoisy))
    show_img("denoisy image", img_denoisy / np.max(img_denoisy))
    print(np.max(img_denoisy))
    plt.show()
