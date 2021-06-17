#introduction implementation yolo
# following along the tutorial:
# https://www.pyimgsearch.com/2018/11/12/yolo-object-detection-with-opencv/
# not done for video but for a sequence of images (frames) instead

import os
import sys
import numpy as np
import argparse
import time
import cv2
import logging as lg

#path variables
darknet_path = "/home/felix/Programs/darknet"  # $DARKNET_PATH

#argument parsing  -> still improvement to be done
ap = argparse.ArgumentParser()
ap.add_argument("-is", "--image_sequence", required=True,
                help="path to input image squence folder to perform detection")

ap.add_argument("-l", "--logging", type=bool, default=False,
                help="enable or disable logging")

args = vars(ap.parse_args())

# set logging stuff
print("Running sequence analysis using YOLO")

if args["logging"]:
    print("Logging is activated - but not implemented yet")
    lg.basicConfig(level=lg.INFO)

#log some basic inputs
lg.info("video stream: {}".format(args["image_sequence"]))

#specify sequence path
image_sequence_folder = os.path.join(args["image_sequence"], 'color')
sequence_images = sorted(os.listdir(image_sequence_folder))
ground_truth_path = os.path.join(args["image_sequence"], "groundtruth.txt")

#parse ground truth
with open(ground_truth_path, 'r') as f:
    gt_list = [[float(n) for n in gt.split(',')] for gt in [l.strip() for l in f]]

lg.info("=======start detection=========")

for frame_number, img_name in enumerate(sequence_images):
    #construct image path and read in img
    img_path = os.path.join(image_sequence_folder, img_name)
    img = cv2.imread(img_path)
    (H, W) = img.shape[:2]
    lg.info("Process frame {}".format(frame_number))

    # draw ground truth box
    [x0, y0, _, _, x2, y2, _, _] = gt_list[frame_number]
    #w, h = int(x2 -x0), int(y2 -y0)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.putText(img, "Ground Truth", (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)

    # show the output image sequence
    cv2.imshow('img', img)
    k = cv2.waitKey(20) & 0xff
    if k == 32:
        break

#cv2.destroyAllWindows()
