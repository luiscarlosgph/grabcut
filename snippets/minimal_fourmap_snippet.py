#!/usr/lib/python
#
# @brief Minimal code snippet to show GrabCut segmentation with a fourmap.
#
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   11 Feb 2021.

import cv2
import grabcut

image_path = 'data/tool_512x409.png' 
fourmap_path = 'data/fourmap_512x409.png'
output_path = 'output_512x409_fourmap_iter_5_gamma_10.png'
max_iter = 5
gamma = 10.

# Read image and trimap
im = cv2.imread(image_path)
im_bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
fourmap = cv2.imread(fourmap_path, cv2.IMREAD_GRAYSCALE)

# Perform segmentation
gc = grabcut.GrabCut(max_iter)
segmentation = gc.estimateSegmentationFromFourmap(im_bgra, fourmap, gamma)

# Save segmentation
cv2.imwrite(output_path, segmentation)
