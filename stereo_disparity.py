# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:32:19 2020

@author: szabo
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import imtools_msz as imsz

crop=True
max_dim=1386

im_id='IMG00000000000000000000.png'

imagePaths=[r'e:\DATA\BRAUN\20201019_test_images\camera_2\IMG00000000000000000000.png',
            r'e:\DATA\BRAUN\20201019_test_images\camera_1\IMG00000000000000000000.png']
            


# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    img = cv2.imread(imagePath,0)
    img, scale = imsz.imRescaleMaxDim(img, max_dim, boUpscale = False, interpolation = 1)
    images.append(img)


imgL = images[0]
imgR = images[1]

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=31)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()