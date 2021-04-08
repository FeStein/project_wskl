import cv2
import numpy as np
import json

#import detection

def calculate_IOU(det1, det2):
    xA = max(det1.x1, det2.x1)
    yA = max(det1.y1, det2.y1)
    xB = min(det1.x2, det2.x2)
    yB = min(det1.y2, det2.y2)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((det1.x2 - det1.x1) * (det1.y2 - det1.y1))
    boxBArea = abs((det2.x2 - det2.x1) * (det2.y2 - det2.y1))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

class Tube():

    """Tube class, which represents an object present on consecutive frames"""

    def __init__(self, detection, id, threshold = 0.6):
        """Initializes a new tube """
        self.detection_list = [] # ordered list of consecutive detections
        self.detection_list.append(detection)
        self.threshold = threshold
        self.label = detection.label
        self.id = id

    def add(self, detection_candidates):
        """Adds a new detection in a new frame to the tube by finding the
        biggest iuo between the last frame of the tube and the list of
        candidates

        :detection_candidates: detections that may be corresponding to the tube
        -----------------------------------------------------------------------
        return: None 

        """
        best_cand = None
        best_iou = 0.0
        last_detection = self.get_last_det()
    
        for cand in detection_candidates: 
            if cand.label == self.label:
                curr_iou = calculate_IOU(last_detection, cand)
                if  curr_iou > best_iou:
                    best_iou = curr_iou 
                    best_cand = cand
        print("Best IoU: {}".format(best_iou))
        if best_iou >= self.threshold:
            self.detection_list.append(best_cand)
            return best_cand
        return None
        
    def is_active(self):
        return True

    def __len__(self):
        return len(self.detection_list)

    def get_last_det(self):
        return self.detection_list[-1]

    def get_first_det(self):
        return self.detection_list[0]

    def get_last_frame(self):
        return self.detection_list[-1].frame_number

    def get_first_frame(self):
        return self.detection_list[0].frame_number

class TubeGenerator():

    """Generator class to keep track of all tubes in a video sequence"""

    def __init__(self, settings_json, detection_list = None):
        """Initializes a new tube generator. Recommended use: initalize it
        before you loop over the video frames """

        with open(settings_json) as settings_file:
            settings = json.load(settings_file)["tube"]

        self.min_tube_length = settings["min_tube_length"]
        self.threshold = settings["threshold"]
        self.break_frames = settings["break_frames"]

        self.current_frame_number = 0
        self.inactive_tube_list = [] # contains all tubes
        self.active_tube_list = [] # contains currently "active" tubes
        
        if detection_list:
            for id, det in enumerate(detection_list):
                tube = Tube(det, id, self.threshold)
                self.active_tube_list.append(tube)


    def update(self, detection_list, skipcounter = False):
        """Updates the current tubes by the detections given. Will initialize
        new tubes when necessary and close unnecessary tubes when not needed.

        :detection_list: List of detections (output of the detection per frame)
        :skipcounter: skip increasing of the frame number -> useful for forcing
        a tube initilization for example via a ground truth bounding box
        -----------------------------------------------------------------------

        """
        #this should be changed to a global number, keeping a separate frame
        #number. keeping separte frame numbers in subclasses is harder to track
        #errors
        if not skipcounter: self.current_frame_number += 1 

        candidates = detection_list.copy()
        
        # add current detections to tubes
        for tube in self.active_tube_list:
            cand = tube.add(candidates)

        #print("working")
        # create new tubes for leftover detections
        #for left_cand in candidates:
        #    new_tube = Tube(left_cand) 
        #    self.active_tube_list.append(new_tube)

        # check for inactive
        #def inactive(tube):
        #    a = tube.get_last_frame() < self.current_frame_number - self.break_frames        
        #    #for tube in self.active_tube_list:
        #    b = len(Tube) >= self.min_tube_length
        #    return a and b

    def output(self):
        print("Num active tubes {}".format(len(self.active_tube_list))) 
        for i,tube in enumerate(self.active_tube_list):
            print("Tube number {}".format(i))
            print("len: {}".format(len(tube)))
            print("last frame {}".format(tube.get_last_frame()))
            print("first frame {}".format(tube.get_first_frame()))
        

    def save(self, filename):
        """Save the detected tubes. Recommended after a detection is finished

        TODO:
        - create useful format
        - create useful output

        :filename: path to outputfile
        :returns: creates a file containing the tubes

        """
        pass


if __name__ == "__main__":
    d1 =  detection.Detection('car', 507, 205, 561, 241, 1)
    d2 =  detection.Detection('traffic light', 139, 105, 151, 121, 1)
    iou = calculate_IOU(d1,d2) 
    print(iou)
    print("no errors")
