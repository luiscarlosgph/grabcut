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
        required = True, 
        help = 'Path to the input RGB image.'
    )
    
    parser.add_argument(
        '--output', 
        required = True, 
        help = 'Path to the output segmentation map.'
    )

    parser.add_argument(
        '--iter', 
        required = True, 
        help = 'Maximum number of GrabCut iteraions.'
    )
    
    parser.add_argument(
        '--gamma', 
        required = True, 
        help = 'GrabCut gamma parameter. Typical value 50.0.'
    )
    
    # Optional parameters
    parser.add_argument(
        '--fourmap', 
        required = False, 
        default = None,
        help = 'Path to the input fourmap.'
    )

    return  parser.parse_args()

##
# @brief Validates the input parameters given by the user in the command line.
#
def validate_cmdline_params(args):
    assert(args.fourmap is not None)

##
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

    # Read fourmap
    raw_fourmap = cv2.imread(args.fourmap) 
    raw_fourmap = raw_fourmap[:, :, 0].astype(np.uint8)

    # Convert fourmap to OpenCV format
    fourmap = np.empty_like(raw_fourmap)
    fourmap[:, :] = cv2.GC_PR_BGD
    fourmap[raw_fourmap == 0] = cv2.GC_BGD
    fourmap[raw_fourmap == 64] = cv2.GC_PR_BGD 
    fourmap[raw_fourmap == 128] = cv2.GC_PR_FGD 
    fourmap[raw_fourmap == 255] = cv2.GC_FGD

    # Create GrabCut algo
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    seg = np.zeros_like(fourmap, dtype = np.uint8)

    # Compute segmentation
    tic = time.time()
    seg, bgd_model, fgd_model = cv2.grabCut(im, fourmap, None, bgd_model, fgd_model, 
        args.iter, cv2.GC_INIT_WITH_MASK)
    toc = time.time() 
    print('Time elapsed in GrabCut segmentation: ' + str(toc - tic)) 

    # Save output
    seg[seg == cv2.GC_BGD] = 0
    seg[seg == cv2.GC_PR_BGD] = 0
    seg[seg == cv2.GC_PR_FGD] = 255
    seg[seg == cv2.GC_FGD] = 255
    cv2.imwrite(args.output, seg)

if __name__ == "__main__":
    main()
