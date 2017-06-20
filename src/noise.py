import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import argparse


parser = argparse.ArgumentParser(
    description="Add the noise to the input image.")
parser.add_argument("-i", "--impath", type=str, required=True,
                    help="input image path")
parser.add_argument("--t", type=str, default='gauss',
                    help="The type of noise.")


def show_img(title, img, cmap=None):
    plt.figure()
    plt.title(title)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)


def add_noise(image, noise_typ, mean=None, var=None, spr=None, amount=None):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        assert mean is not None and var is not None
        mean = mean
        var = var
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = np.abs(gauss.reshape(row, col, ch))
        noisy = image + gauss
        return noisy
    elif noise_typ == "snp":
        row, col, ch = image.shape
        assert spr is not None and amount is not None
        s_vs_p = spr
        amount = amount
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0
        return noisy


if __name__ == '__main__':
    noise_type = ['gauss', 'snp']
    args = parser.parse_args()
    if args.t not in noise_type:
        print("There are only {} in noise type.".format(noise_type))
    img = (skimage.io.imread(args.impath).astype(np.float32))
    img_noisy = add_noise(img, args.t)
    plt.show()
