import os
import sys
import numpy as np
import cv2
import logging as lg

import json

import SpatioTemporal.detection as det
import SpatioTemporal.tube as tb

with open("settings.json") as config_file:
    settings = json.load(config_file)

# set logging stuff
print("Running sequence analysis using YOLO")

if settings["general"]["logging"]:
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
frame_number = 0 #oth frame is init  
init_det_list = [det.Detection("car", 246,162,357,279, frame_number)]

TG = tb.TubeGenerator("settings.json", init_det_list)

for frame_number, img_name in enumerate(sequence_images):
    frame_number += 1

    #construct image path and read in img
    img_path = os.path.join(image_sequence_folder, img_name)

    img = cv2.imread(img_path)

    lg.info("Process frame {}".format(frame_number))

    # detect objects
    detections = YDetect.detect(img, frame_number)


    #set gt as detection
    x1,y1,_,_,x2,y2,_,_ = gt_list[frame_number -1]
    dett = det.Detection("car",int(x1),int(y1),int(x2),int(y2),frame_number)
    detections = [dett]
    
    for dett in detections:
        if dett.label == "truck" or dett.label == "bus":
            dett.label = "car"

    TG.update(detections)

    VIS.visualize(detections,img)

TG.output()

VIS.destruct()
