import argparse
import sys
import dehaze
import cv2


parser = argparse.ArgumentParser(description="Implement Recover Haze image.")
parser.add_argument("-i", "--impath", type=str, required=True,
                    help="input image path")
parser.add_argument("-o", "--out", type=str, default="out.jpg",
                    help="output image path (default: out.jpg)")
parser.add_argument("-s", "--patch_size", type=int, default=15)
parser.add_argument("-p", "--top_portion", type=float, default=0.001)
parser.add_argument("--t0", type=float, default=0.1)
parser.add_argument("--omega", type=float, default=0.95)


def main(args):
    img = cv2.imread(args.impath)
    print(img.shape)
    img_dehaze = dehaze.dehaze(
        img, args.patch_size, args.top_portion, args.t0, args.omega)
    cv2.imwrite(args.out, img_dehaze)

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
    # if args.k <= 0:
    #     print("window size should be positive.")
    main(args)
