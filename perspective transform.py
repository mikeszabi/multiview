# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:44:38 2020

@author: szabo
"""


import numpy as np
import cv2


import imtools_msz as imsz

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type=str, required=True,
# 	help="path to input directory of images to stitch")
# ap.add_argument("-o", "--output", type=str, required=True,
# 	help="path to the output image")
# ap.add_argument("-c", "--crop", type=int, default=0,
# 	help="whether to crop out largest rectangular region")
# args = vars(ap.parse_args())

def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = 0
    y = 0
    while x < img.shape[1]+pxstep:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]+pxstep:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep


crop=True
max_dim=1386


# Base nest image: 4x1
img_base = np.zeros([200,400,3],dtype=np.uint8)
img_base.fill(0) 
draw_grid(img_base, line_color=(0, 255, 0), thickness=3, type_=cv2.LINE_AA, pxstep=100)

nest_points_base=[[0,0],[100,0],[200,0],[300,0],[400,0],[400,200],[300,200],[200,200],[100,200],[0,200]]
nest_points=[None]*4

M=[None]*4


#### Load images


imagePaths=[r'e:\DATA\BRAUN\20201019_test_images\camera_2\IMG00000000000000000000.png',
            r'e:\DATA\BRAUN\20201019_test_images\camera_1\IMG00000000000000000000.png',
            r'e:\DATA\BRAUN\20201019_test_images\camera_3\IMG00000000000000000000.png',
            r'e:\DATA\BRAUN\20201019_test_images\camera_0\IMG00000000000000000000.png']


# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    img, scale = imsz.imRescaleMaxDim(img, max_dim, boUpscale = False, interpolation = 1)
    images.append(img)



###########################

img = images[0]

draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=25)


nest_points[0]=[[500,175],[815,175],None,None,None,None,None,None,[815,805],[500,805]]

npb_points=[npb for npb, np in zip(nest_points_base,nest_points[0]) if np is not None]
np_points=[np for npb, np in zip(nest_points_base,nest_points[0]) if np is not None]

pts1 = np.float32(npb_points)
pts2 = np.float32(np_points)

M[0] = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img_base,M[0],img.shape[1::-1])

pts = np.array([pts1])
cv2.perspectiveTransform(pts, M[0])


added_image = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),0.7,dst,0.3,0)

plt.imshow(added_image)

#############################

img = images[1]

draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=25)


nest_points[1]=[[300,175],[615,175],None,None,None,None,None,None,[615,805],[300,805]]

npb_points=[npb for npb, np in zip(nest_points_base,nest_points[1]) if np is not None]
np_points=[np for npb, np in zip(nest_points_base,nest_points[1]) if np is not None]

pts1 = np.float32(npb_points)
pts2 = np.float32(np_points)

M[1] = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img_base,M[1],img.shape[1::-1])

pts = np.array([pts1])
cv2.perspectiveTransform(pts, M[1])

added_image = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),0.7,dst,0.3,0)

plt.imshow(added_image)

##########################


img = images[2]

draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=25)


nest_points[2]=[None,None,None,[785,175],[1100,175],[1100,805],[785,805],None,None,None]

npb_points=[npb for npb, np in zip(nest_points_base,nest_points[2]) if np is not None]
np_points=[np for npb, np in zip(nest_points_base,nest_points[2]) if np is not None]

pts1 = np.float32(npb_points)
pts2 = np.float32(np_points)

M[2] = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img_base,M[2],img.shape[1::-1])

pts = np.array([pts1])
cv2.perspectiveTransform(pts, M[2])

added_image = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),0.7,dst,0.3,0)

plt.imshow(added_image)

##########################


img = images[3]

draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=25)


nest_points[3]=[None,None,None,[540,175],[855,175],[855,805],[540,805],None,None,None]

npb_points=[npb for npb, np in zip(nest_points_base,nest_points[3]) if np is not None]
np_points=[np for npb, np in zip(nest_points_base,nest_points[3]) if np is not None]

pts1 = np.float32(npb_points)
pts2 = np.float32(np_points)

M[3] = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img_base,M[3],img.shape[1::-1])

pts = np.array([pts1])
cv2.perspectiveTransform(pts, M[3])

added_image = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),0.7,dst,0.3,0)

plt.imshow(added_image)

####################
base_dir=r'e:\DATA\BRAUN\kalib_cam1-4'
proj_matrix_file=os.path.join(base_dir,'projections')
np.savez(proj_matrix_file, cam1=M[0], cam2=M[1], cam3=M[2], cam4=M[3])
