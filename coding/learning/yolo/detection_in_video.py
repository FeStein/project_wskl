#introduction implementation yolo
# following along the tutorial:
# https://www.pyimgsearch.com/2018/11/12/yolo-object-detection-with-opencv/

import numpy as np
import argparse
import time
import cv2
import os
import logging

#path variables
darknet_path = "/home/felix/Programs/darknet"  # $DARKNET_PATH

#argument parsing  -> still improvement to be done
ap = argparse.ArgumentParser()
ap.add_argument("-vs", "--video_stream", required=True,
                help="path to input img to perform detection")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")

ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")

ap.add_argument("-l", "--logging", type=bool, default=False, 
                help="enable or disable logging")

args = vars(ap.parse_args())

print("Running video analysis using YOLO")

if args["logging"]: print("Logging is activated - but not implemented yet")

#get labeles -> contained in darknet data folder | all labels yolo is trained on
labelsPath = os.path.sep.join([darknet_path, "data", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# create random color set for uniqe coloring of bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# load weights and config from pre-trained model
weightsPath = os.path.sep.join([darknet_path, "yolov3.weights"])
configPath = os.path.sep.join([darknet_path, "cfg", "yolov3.cfg"])

# load YOLO from its path
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#get names of output layer from yolo
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# get video stream tbd: via argparse
cap = cv2.VideoCapture(args["video_stream"])

while True:
    ret, img = cap.read()
    (H, W) = img.shape[:2]

    #blop -> forward pass to YOLO obj. detector: bounding boxes + probabilities
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("Frame analysis via YOLO took {:.6f} seconds".format(end - start))

    #visualize results:
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence of bounding box
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter bad predictions and calulate box from center stuf
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():

            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the img
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output video stream
    cv2.imshow('img', img)
    k = cv2.waitKey(20) & 0xff
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()
