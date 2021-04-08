import abc
import cv2
import os
import json

import sys
import numpy as np
import time
import logging as lg

class Detection():

    """Class containing a single detection result"""

    def __init__(self, label, x1, y1, x2, y2, frame_number):
        self.label = label
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number #number of the frame detection belongs to

    def __str__(self):
        return " ".join([str(i) for i in[self.x1, self.y1, self.x2, self.y2]])

class Detector(abc.ABC):
    
    """Abstract Detector class, defining the basiscs of a object detector"""

    def __init__(self, settings_json):
        """TODO: to be defined. """

        with open(settings_json) as settings_file:
            settings = json.load(settings_file)["detection"]
        
        self.labelPath = settings["labelPath"]
        self.LABLES = open(self.labelPath).read().strip().split("\n")
        self.weightsPath = settings["weightsPath"]
        self.configPath = settings["configPath"]

        self.logging = settings["logging"]
        self._init_logging(settings["logLevel"])

    def _init_logging(self, log_level):
        """Init the logging for object detection"""
        lg.basicConfig(format="ObjectDetector - %(levelname)s - %(message)s", level = log_level)
    
    @abc.abstractmethod
    def detect(self, image, frame_number):
        """Detects the bounding boxes given an image

        :image: image in opencv format
        :returns: List of bounding boxes in format:
            [(label, (x,y), (x + w,y + h])]
        """
        return



class YOLO_Detector(Detector):

    """Object detector using the YOLO Algorithm"""

    def __init__(self, settings_json):
        """TODO: to be defined. """
        Detector.__init__(self, settings_json)
        lg.info("Initialize YOLO Object detector")

        with open(settings_json) as settings_file:
            settings = json.load(settings_file)["detection"]

        self.darknetPath = settings["darknetPath"]
        self.confidence = settings["confidence"]
        self.threshold = settings["threshold"]

        # load YOLO from its path
        lg.debug("Load YOLO weigths")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

        #get names of output layer from yolo
        lg.debug("Setting layer names")
        self.layerNames = self.net.getLayerNames()
        self.layerNames = [self.layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.image_number = 0



    def detect(self, image, frame_number):
        #blop -> forward pass to YOLO obj. detector: bounding boxes + probabilities
        self.image_number += 1
        start = time.time()
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.layerNames)
        end = time.time()
        lg.info( "Object detection in Image {} via YOLO took {:.7f} seconds".format(self.image_number, end - start))

        (H, W) = image.shape[:2]

        boxes = []
        confidences = []
        classIDs = []
        
        lg.debug("{} layer outputs found".format(len(layerOutputs)))
        for output in layerOutputs:
            for detection in output:
                # extract the class ID and confidence of bounding box
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                #filter bad predictions and calulate box from center stuf
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
                self.threshold)

        lg.debug("{} resulting bounding boxes found".format(len(idxs)))
        
        detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                det = Detection(self.LABLES[classIDs[i]], x, y, x + w, y + h, frame_number)
                detections.append(det)

        return detections

class FRCNN_Detector(Detector):

    """Object detector using the Faster R-CNN Algorithm"""

    def __init__(self):
        """TODO: to be defined. """
        Detector.__init__(self)
        pass # will be implemented later

    def detect(self, image):
        pass

class Visualizer():

    """Visualizes bounding box output"""

    def __init__(self,settings_json):
        """TODO: to be defined.

        :settings: TODO

        """
        print("Init vis")

    def visualize(self, detections, image, color = 123):

        for bb in detections:
            cv2.rectangle(image, (bb.x1, bb.y1), (bb.x2, bb.y2), color, 2)
            text = bb.label 
            cv2.putText(image, bb.label, (bb.x2, bb.y2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        cv2.imshow('img', image)
        k = cv2.waitKey(20) & 0xff
        if k == 32:
            cv2.destroyAllWindows()
            sys.exit()

    def destruct(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    det = YOLO_Detector("../settings.json")
    print("no errors")
