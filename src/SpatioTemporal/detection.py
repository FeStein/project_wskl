import abc
import cv2
import os
import json

import sys
import numpy as np
import argparse
import time
import logging as lg

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
        self._init_logging()

    def _init_logging(self):
        """Init the logging for object detection"""
        pass
    
    @abc.abstractmethod
    def detect(self, image):
        """Detects the bounding boxes given an image

        :image: image in opencv format
        :returns: List of bounding boxes in format:
            [(label, [x,y,h,w])]
        """
        return



class YOLO_Detector(Detector):

    """Object detector using the YOLO Algorithm"""

    def __init__(self, settings_json):
        """TODO: to be defined. """
        Detector.__init__(self, settings_json)

        with open(settings_json) as settings_file:
            settings = json.load(settings_file)["detection"]

        self.darknetPath = settings["darknetPath"]

        # load YOLO from its path
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

        #get names of output layer from yolo
        self.layerNames = self.net.getLayerNames()
        self.layerNames = [self.layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]



    def detect(self, image):
        #blop -> forward pass to YOLO obj. detector: bounding boxes + probabilities
        start = time.time()
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.layerNames)
        end = time.time()

        lg.info( "Object detection in Frame via YOLO took {:.7f} seconds".format(end - start))

        return layerOutputs

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
        self._settings = settings

    def visualize(self):
        pass 

if __name__ == "__main__":
    det = YOLO_Detector("../settings.json")
    print("no errors")
