# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:57:14 2020

@author: szabo
"""
import os
import cv2
import imtools_msz as imsz
(major, minor) = cv2.__version__.split(".")[:2]

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
    
choice='medianflow'
tracker = OPENCV_OBJECT_TRACKERS[choice]()

initBB = None

#################
base_dir=r'd:\DATA\EON_LOCAL\SESSIONS\20200218_092900'
#run_id='20180907_103336_sel1'
run_id='20200218_092854964_Agent4_Master'
ext='.jpg'
image_max_dim=1014


im_dir=os.path.join(base_dir,run_id)
jpg_list=imsz.imagelist_in_depth(im_dir,level=1)

frame_id=0
wait_time=100
while True:
    frame = cv2.imread(jpg_list[frame_id],0)  #queryimage # left image
    frame, scale = imsz.imRescaleMaxDim(frame, image_max_dim, boUpscale = False, interpolation = 1)

    (H, W) = frame.shape[:2]
    frame_id+=1
    if frame_id>len(jpg_list):
        break

# check to see if we are currently tracking an object
    if initBB is not None:
    # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
            
        else:
            initBB=None
    
        info = [
        ("Tracker", choice),
        ("Success", "Yes" if success else "No"),
        ("Frame", "{:d}".format(frame_id)),
        ]
# loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(wait_time) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
    if key == ord("d"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
        initBB=None
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        wait_time=0
        
    if key == ord("s"):
        if wait_time>0:
            wait_time=0
        else:
            wait_time=100
        
        
# if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()