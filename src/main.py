import argparse
import sys
import dehaze
import numpy as np
import skimage.io
import os
import noise

parser = argparse.ArgumentParser(description="Implement Recover Haze image.")
parser.add_argument("-o", "--out", type=str, default="../result/out",
                    help="output image dir (default: ../result/out")
parser.add_argument("-s", "--patch_size", type=int,
                    default=15, help='local patch size (default: 15)')
parser.add_argument("-p", "--top_portion", type=float, default=0.001,
                    help='top atmosphere pixel portion (default:0.001)')
parser.add_argument("--t0", type=float, default=0.1,
                    help='minimum transmission rate(0-1) (default: 0.1)')
parser.add_argument("--omega", type=float, default=0.95,
                    help='natural transmission constant(0-1) (default: 0.95)')
parser.add_argument("-i", "--impath", type=str, required=True,
                    help="input image path")
parser.add_argument("-n", "--noise_type", type=str,
                    default=None, help="noise type(gauss|s&p)")
parser.add_argument("--mean", type=float, default=None,
                    help="Gaussian noise mean")
parser.add_argument("--var", type=float, default=None,
                    help="Gaussian noise var")
parser.add_argument("--ratio", type=float, default=None,
                    help="Salt and Pepper ratio")
parser.add_argument("--amount", type=float, default=None,
                    help="Salt and Pepper amount")


def main(args):
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    img = (skimage.io.imread(args.impath).astype(np.float32))
    if args.noise_type is not None:
        img = noise.add_noise(img, args.noise_type, mean=args.mean,
                              var=args.var, spr=args.ratio, amount=args.amount)

    img_dehaze = dehaze.dehaze(
        img, args.patch_size, args.top_portion, args.t0, args.omega, args.out)
    skimage.io.imsave(os.path.join(args.out, 'equalize.jpg'), img_dehaze)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.patch_size <= 0:
        print("patch size should be positive.")
        sys.exit()
    if args.top_portion <= 0 or args.top_portion >= 1:
        print("Top portion should be in (0,1).")
        sys.exit()
    if args.omega <= 0 or args.omega >= 1:
        print("Omega should be in (0,1).")
        sys.exit()
    main(args)
