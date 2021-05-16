import os
import sys
import numpy as np
import cv2
import logging as lg

import json

import SpatioTemporal.detection as det
import SpatioTemporal.tube as tb

from collections import Counter
from tqdm import tqdm

with open("settings.json") as config_file:
    settings = json.load(config_file)

if settings["general"]["logging"]:
    lg.basicConfig(level=lg.INFO)

#log some basic inputs
lg.info("video stream: {}".format(settings["path"]["images"]))

sequence_path = "/home/felix/Documents/vot2019/sequences"
sequence_names = sorted(os.listdir(sequence_path))

#sequence_names = ["car1"]

csv_line = ["folder_name","total_frames","accuracy","skipped_frames","first_label","first_occ","second_label", "second_occ", "third_label", "third_occ"]
with open("../logs/accuracy_YOLO.log",'w+') as f:
    f.write(",".join(csv_line) + "\n")

for sname in tqdm(sequence_names):
    if sname == "flamingo1":
        continue

    print("Current sequence: {}".format(sname))

    #specify sequence path
    image_sequence_folder = os.path.join("/home/felix/Documents/vot2019/sequences", sname, "color")
    sequence_images = sorted(os.listdir(image_sequence_folder))
    ground_truth_path = os.path.join(settings["path"]["images"], settings["path"]["ground_truth_name"])

    #parse ground truth
    with open(ground_truth_path, 'r') as f:
        gt_list = [[float(n) for n in gt.split(',')] for gt in [l.strip() for l in f]]
    
    #init object detector
    YDetect = det.YOLO_Detector("settings.json")
    
    lg.info("=======start detection=========")
    
    frame_number = 0 #0th frame is init  
    x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[0]
    #calculate rectangular ground truth
    xx1 = min(x1,x2,x3,x4)
    xx2 = max(x1,x2,x3,x4)
    yy1 = min(y1,y2,y3,y4)
    yy2 = max(y1,y2,y3,y4)
    
    init_det_list = [det.Detection("car", int(xx1), int(yy1), int(xx2), int(yy2), frame_number)]
    
    accuracry_list = []
    label_list = []
    number_no_detection = 0
    
    for frame_number, img_name in tqdm(enumerate(sequence_images)):
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
    
        ################################################################################
            # Calculate Accuracy
        ################################################################################
        
        #create ground truth detection
        x1,y1,x2,y2,x3,y3,x4,y4 = gt_list[frame_number -1]
        xx1 = min(x1,x2,x3,x4)
        xx2 = max(x1,x2,x3,x4)
        yy1 = min(y1,y2,y3,y4)
        yy2 = max(y1,y2,y3,y4)
    
        gt_dett = det.Detection("groundTruth",int(xx1),int(yy1),int(xx2),int(yy2),frame_number)
    
        #find best matching IoU
        best_iou = 0.0
        best_label = "dummy"
        for trial_det in detections:
            curr_iou = tb.calculate_IOU(trial_det, gt_dett)
            if curr_iou >= best_iou:
                best_iou, best_label = curr_iou, trial_det.label
    
        if best_iou == 0.0:
            number_no_detection +=1
    
        accuracry_list.append(best_iou)
        label_list.append(best_label)
    
        ################################################################################
        
    
        #visualize loaded tube
        if False:
            vis_det = []
            for tube in TG_loaded.active_tube_list:
                for detec in tube.detection_list:
                    if detec.frame_number == frame_number:
                        detec.label = str(tube.id)
                        vis_det.append(detec)
            VIS.visualize(vis_det,img,color=(0,255,0))
    
        if False:
            #visualize ground turth
            detections = [dett]
            pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,255),2)
            VIS.visualize(detections,img, color=(0,0,255))
            cv2.imwrite(settings["path"]["output"] + "img_{}.png".format(frame_number),img)
    
    #calc overall accuracy
    acc = sum(accuracry_list) / len(accuracry_list)
    most_comm_labels = Counter(label_list).most_common(3)
    mc_list = []
    for mc in most_comm_labels:
        mc_list.append(str(mc[0]))
        mc_list.append(str(mc[1]))
    
    csv_line = [sname,str(len(sequence_images)),str(acc) ,str(number_no_detection)] + mc_list

    #print csv_list
    with open("../logs/accuracy_YOLO.log",'a') as f:
        f.write(",".join(csv_line) + "\n")
