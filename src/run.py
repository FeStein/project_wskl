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

for frame_number, img_name in enumerate(sequence_images[:20]):
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
    detections = [det for det in detections if det.label == "car"] 

    TG.update(detections)

    # visualize current tube in frame
    #vis_det = []
    #lookup_list = TG.active_tube_list.copy()
    #for tube in lookup_list:
    #    last = tube.get_last_det()
    #    if last.frame_number == frame_number:
    #        print(last.frame_number)
    #        last.label = str(tube.id)
    #        vis_det.append(last)
    #VIS.visualize(vis_det,img,color=(255,0,0))


    #visualize loaded tube
    vis_det = []
    for tube in TG_loaded.active_tube_list:
        for detec in tube.detection_list:
            if detec.frame_number == frame_number:
                detec.label = str(tube.id)
                vis_det.append(detec)
    VIS.visualize(vis_det,img,color=(0,255,0))

    #visualize ground turth
    x1,y1,_,_,x2,y2,_,_ = gt_list[frame_number -1]
    dett = det.Detection("GT",int(x1),int(y1),int(x2),int(y2),frame_number)
    detections = [dett]
    VIS.visualize(detections,img, color=(0,0,255))
    cv2.imwrite(settings["path"]["output"] + "img_{}.png".format(frame_number),img)

    #if len(vis_det) != 0:
    #    iou = tb.calculate_IOU(vis_det[-1],dett)
    #    print('Ground Truth IoU:{}'.format(iou))


TG.finish()

TG.save(settings["path"]["output"] + "")

TG.output()

VIS.destruct()
