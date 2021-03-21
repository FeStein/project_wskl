import os
import sys
import numpy as np
import cv2
import logging as lg

import json

import SpatioTemporal.detection as det

with open("settings.json") as config_file:
    settings = json.load(config_file)

# set logging stuff
print("Running sequence analysis using YOLO")

if settings["general"]["logging"]:
    print("Logging is activated - but not implemented yet")
    lg.basicConfig(level=lg.INFO)

#log some basic inputs
lg.info("video stream: {}".format(settings["path"]["images"]))

#specify sequence path
image_sequence_folder = os.path.join(settings["path"]["images"], "color")
sequence_images = sorted(os.listdir(image_sequence_folder))
ground_truth_path = os.path.join(settings["path"]["images"], settings["path"]["ground_truth_name"])

#parse ground truth
with open(ground_truth_path, 'r') as f:
    gt_list = [[float(n) for n in gt.split(',')] for gt in [l.strip() for l in f]]

#init object detector
YDetect = det.YOLO_Detector("settings.json")

lg.info("=======start detection=========")

VIS = det.Visualizer("settings.json")

for frame_number, img_name in enumerate(sequence_images):
    #construct image path and read in img
    img_path = os.path.join(image_sequence_folder, img_name)
    img = cv2.imread(img_path)
    lg.info("Process frame {}".format(frame_number))

    # detect objects
    detections = YDetect.detect(img)
    
    VIS.visualize(detections,img)

VIS.destruct()
