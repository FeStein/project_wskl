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
frame_number = 0 #0th frame is init  
x1,y1,_,_,x2,y2,_,_ = gt_list[0]
init_det_list = [det.Detection("car", int(x1), int(y1), int(x2), int(y2), frame_number)]

TG = tb.TubeGenerator("settings.json", init_det_list)

TG_loaded = tb.TubeGenerator("settings.json")
TG_loaded.load(settings["path"]["output"])

TG_loaded.output()

past_dett = None

for frame_number, img_name in enumerate(sequence_images):
    frame_number += 1

    #construct image path and read in img
    img_path = os.path.join(image_sequence_folder, img_name)

    img = cv2.imread(img_path)

    lg.info("Process frame {}".format(frame_number))

    # detect objects
    detections = YDetect.detect(img, frame_number)
    # set detection of truck or bus equal to car (simplification since I don't
    # want to custom train a network) 
    for dett in detections:
        if dett.label == "truck" or dett.label == "bus":
            dett.label = "car"

    # filter for car detections (just debugging)
    #detections = [det for det in detections if det.label == "car"] 



    #visualize ground turth
    x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[frame_number -1]
    #calculate rectangular ground truth
    xx1 = min(x1,x2,x3,x4)
    xx2 = max(x1,x2,x3,x4)
    yy1 = min(y1,y2,y3,y4)
    yy2 = max(y1,y2,y3,y4)

    dett = det.Detection("",int(xx1),int(yy1),int(xx2),int(yy2),frame_number)
    ydetections = [dett]
    pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
    pts = pts.reshape((-1,1,2))
    #cv2.polylines(img,[pts],True,(0,255,255),2)
    VIS.visualize(ydetections,img, color=(0,0,255))

    gt_dett = det.Detection("groundTruth",int(xx1),int(yy1),int(xx2),int(yy2),frame_number)

    #find best matching IoU
    best_iou = 0.0
    best_label = "YOLO"
    best_det = None
    for trial_det in detections:
        curr_iou = tb.calculate_IOU(trial_det, gt_dett)
        if curr_iou >= best_iou:
            best_iou, best_label, best_det = curr_iou, trial_det.label, trial_det

    print(len(detections))
    if best_iou != 0.0 and best_det:
        print(best_label)
        VIS.visualize([best_det],img, color=(255,255,255))

    cv2.imwrite(settings["path"]["output"] + "img_{}.png".format(frame_number),img)

TG.finish()

TG.save(settings["path"]["output"] + "")

TG.output()

VIS.destruct()
