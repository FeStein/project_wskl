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

################################################################################
track_label = "person"
start_frame = 0
################################################################################

frame_number = 0 #0th frame is init  
x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[0]
#calculate rectangular ground truth
xx1 = min(x1,x2,x3,x4)
xx2 = max(x1,x2,x3,x4)
yy1 = min(y1,y2,y3,y4)
yy2 = max(y1,y2,y3,y4)

init_det_list = [det.Detection(track_label, int(xx1), int(yy1), int(xx2), int(yy2), frame_number)]

accuracry_list = []
label_list = []
number_no_detection = 0

#init tube generator using the initial ground truth
TG = tb.TubeGenerator("settings.json", init_det_list)

TG_loaded = tb.TubeGenerator("settings.json")
TG_loaded.load(settings["path"]["output"])

TG_loaded.output()


frame_number = start_frame
for img_name in sequence_images[start_frame:]:
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

    TG.update(detections)

    #visualize ground turth
    x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[frame_number -1]
    #calculate rectangular ground truth
    xx1 = min(x1,x2,x3,x4)
    xx2 = max(x1,x2,x3,x4)
    yy1 = min(y1,y2,y3,y4)
    yy2 = max(y1,y2,y3,y4)

    dett = det.Detection(track_label,int(xx1),int(yy1),int(xx2),int(yy2),frame_number)
    detections = [dett]
    pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
    pts = pts.reshape((-1,1,2))
    #cv2.polylines(img,[pts],True,(0,255,255),2)
    VIS.visualize(detections,img, color=(0,0,255))

    #check if tube nees to be reenit (lose track)
    if len(TG.active_tube_list) == 0:
        new_tube = tb.Tube(dett, TG.id_counter)
        TG.id_counter += 1
        TG.active_tube_list.append(new_tube)
        print("---------------REINIT------------------")
        print("No tubes found")
        print("---------------REINIT------------------")
    elif tb.calculate_IOU(dett,TG.active_tube_list[0].get_last_det()) == 0.0 and TG.active_tube_list[0].get_last_det().frame_number ==  frame_number:
        TG.inactive_tube_list.append(TG.active_tube_list[0])
        TG.active_tube_list = []
        new_tube = tb.Tube(dett, TG.id_counter)
        TG.id_counter += 1
        TG.active_tube_list.append(new_tube)
        print("---------------REINIT------------------")
        print("No IoU of last frame")
        print("---------------REINIT------------------")

    print("Process frame {}".format(frame_number))
    print("Active tubes: {}".format(len(TG.active_tube_list)))

    print("Last active frame")
    for tube in TG.active_tube_list:
        print(tube.get_last_det().frame_number)
    
    # visualize current tube in frame
    vis_det = []
    lookup_list = TG.active_tube_list.copy()
    for tube in lookup_list:
        last = tube.get_last_det()

        print("Last one".format(last.frame_number))
        if last.frame_number == frame_number:
            print(last.frame_number)
            last.label = str(tube.id)
            vis_det.append(last)

    VIS.visualize(vis_det,img,color=(256,0,0))

      
    #visualize loaded tube
    if False:
        vis_det = []
        for tube in TG_loaded.active_tube_list:
            for detec in tube.detection_list:
                if detec.frame_number == frame_number:
                    detec.label = str(tube.id)
                    vis_det.append(detec)
        VIS.visualize(vis_det,img,color=(0,255,0))

    cv2.imwrite(settings["path"]["output"] + "img_{}.png".format(frame_number),img)



TG.finish()

TG.save(settings["path"]["output"] + "")

TG.output()

accuracy_list = []

num_tubes = 0
num_detections = 0
# tube evaluation
for tube in TG.active_tube_list + TG.inactive_tube_list:
    num_tubes += 1
    for dt in tube.detection_list:
        num_detections += 1
        frame_number = dt.frame_number

        x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[frame_number -1]

        #calculate rectangular ground truth
        xx1 = min(x1,x2,x3,x4)
        xx2 = max(x1,x2,x3,x4)
        yy1 = min(y1,y2,y3,y4)
        yy2 = max(y1,y2,y3,y4)

        gt_det = det.Detection(track_label,int(xx1),int(yy1),int(xx2),int(yy2),frame_number)
        
        print(type(gt_det))
        print(type(dt))
        iou = tb.calculate_IOU(gt_det, dt)

        accuracy_list.append(iou)

accuracy = sum(accuracy_list) / num_detections
video_length = len(sequence_images)
total_accuracy = sum(accuracy_list) / video_length

print("Accuracy: {}".format(accuracy))
print("Robustness: {}".format(num_tubes))
print("No detection: {}".format(video_length - num_detections))
print("Total Accuracy: {}".format(total_accuracy))

VIS.destruct()

