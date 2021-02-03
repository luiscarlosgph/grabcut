#!/usr/lib/python
#
# @brief This is a testing example to use the grabcut Python module. There is a comparison with
#        the OpenCV vesion.
#
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   3 Sep 2019.

import numpy as np
import cv2
import time
import argparse

# My imports
import grabcut


def parse_cmdline_params(parser):

    # Mandatory parameters
    parser.add_argument(
        '--image',
        required=True,
        help='Path to the input RGB image.'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to the output segmentation map.'
    )

    parser.add_argument(
        '--iter',
        required=True,
        help='Maximum number of GrabCut iteraions.'
    )

    parser.add_argument(
        '--gamma',
        required=True,
        help='GrabCut gamma parameter. Typical value 50.0.'
    )

    # Optional parameters
    parser.add_argument(
        '--trimap',
        required=False,
        default=None,
        help='Path to the input trimap.'
    )

    parser.add_argument(
        '--fourmap',
        required=False,
        default=None,
        help='Path to the input fourmap.'
    )

    parser.add_argument(
        '--fgmap',
        required=False,
        default=None,
        help='Path to the foreground probability map. [0, 255] will be mapped to [0, 1].'
    )

    parser.add_argument(
        '--bgmap',
        required=False,
        default=None,
        help='Path to the background probability map. [0, 255] will be mapped to [0, 1].'
    )

    return parser.parse_args()

#
# @brief Validates the input parameters given by the user in the command line.
#


def validate_cmdline_params(args):
    assert(args.trimap or args.fourmap or args.fgmap)

#
# @brief Converts the command line arguments to the right data types.
#


def convert_cmdline_params(args):
    args.iter = int(args.iter)
    args.gamma = float(args.gamma)


def main():
    # Parse command line parameters
    parser = argparse.ArgumentParser()
    args = parse_cmdline_params(parser)
    validate_cmdline_params(args)
    convert_cmdline_params(args)

    # Read image
    im = cv2.imread(args.image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)

    # Create GrabCut algo
    gc = grabcut.GrabCut(args.iter)

    # Compute segmentation
    seg = None
    if args.trimap is not None:
        # Read trimap
        trimap = cv2.imread(args.trimap)[:, :, 0]

        # Compute segmentation
        tic = time.time()
        seg = gc.estimateSegmentationFromTrimap(im, trimap, args.gamma)
        toc = time.time()

    elif args.fourmap is not None:
        # Read fourmap
        fourmap = cv2.imread(args.fourmap)[:, :, 0]

        # Compute segmentation
        tic = time.time()
        seg = gc.estimateSegmentationFromFourmap(im, fourmap, args.gamma)
        toc = time.time()

    elif args.fgmap is not None:
        # Read foreground map
        fgmap = cv2.imread(args.fgmap)[:, :, 0].astype(np.float32) / 255.0
        
        # Read or compute background map
        bgmap = None
        if args.bgmap is not None:
            bgmap = cv2.imread(args.bgmap)[:, :, 0].astype(np.float32) / 255.0
        else:
            bgmap = 1.0 - fgmap;

        # Compute segmentation 
        tic = time.time()
        seg = gc.estimateSegmentationFromProba(im, bgmap, fgmap, args.gamma)
        toc = time.time()

    # Save output and print timing
    seg *= 255  # the output from GrabCut is (0, 1)
    cv2.imwrite(args.output, seg)
    print('Time elapsed in GrabCut segmentation: ' + str(toc - tic))

if __name__ == "__main__":
    main()
