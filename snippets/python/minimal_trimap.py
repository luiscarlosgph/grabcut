#!/usr/lib/python
#
# @brief Minimal code snippet to show GrabCut segmentation with a trimap.
#
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   11 Feb 2021.

import cv2
import grabcut

image_path = 'data/tool_512x409.png' 
trimap_path = 'data/trimap_512x409.png'
output_path = 'data/output_512x409_trimap_iter_5_gamma_10.png'
max_iter = 5
gamma = 10.

# Read image and trimap
im = cv2.imread(image_path)
im_bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

# Perform segmentation
gc = grabcut.GrabCut(max_iter)
segmentation = gc.estimateSegmentationFromTrimap(im_bgra, trimap, gamma)

# Save segmentation
cv2.imwrite(output_path, segmentation)
