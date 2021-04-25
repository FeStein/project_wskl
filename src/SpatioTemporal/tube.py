import cv2
import numpy as np
import json
import os 

import SpatioTemporal.detection as det

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

    def interpolate(self, current_frame_number):
        """
        Interpolates bounding boxes if detection frmames are missing.

        current_frame_number: Number of the current frame in the running detection
        """
        # check if a detection was added in this frame -> makes no sense otherwise
        if self.get_last_frame() != current_frame_number:
            return
        
        start_frame_number = self.detection_list[-2].frame_number
        ds = self.detection_list[-2]
        end_frame_number = self.detection_list[-1].frame_number
        de = self.detection_list[-1]
        # check if frames are missing -> if none missing break
        if  start_frame_number + 1 == end_frame_number:
            return

        # interpolate over consecutive frames (linear)
        num_interpolate = end_frame_number - start_frame_number - 1

        #step size
        xs_s = (de.x1 - ds.x1) / (num_interpolate + 1)
        xe_s = (de.x2 - ds.x2) / (num_interpolate + 1) 
        ys_s = (de.y1 - ds.y1) / (num_interpolate + 1)
        ye_s = (de.y2 - ds.y2) / (num_interpolate + 1) 

        for i in range(num_interpolate):
            xi1 = int(ds.x1 + xs_s * (i + 1))
            xi2 = int(ds.x2 + xe_s * (i + 1))
            yi1 = int(ds.y1 + ys_s * (i + 1))
            yi2 = int(ds.y2 + ye_s * (i + 1))
            di = det.Detection(ds.label, xi1, yi1, xi2 ,yi2, start_frame_number + i + 1, interpolated = True)
            self.detection_list.insert(len(self.detection_list) - 1, di)

    def extrapolate(self, current_frame_number):
        """
        Extrapolates a bounding box to the current frame if 2 consecutive
        frames (curr_frame number - 1, curr_frame - 2) are given.

        current_frame_number: Number of the current frame in the running detection
        """
        # check if a detection was added in this frame -> makes no sense otherwise
        if self.get_last_frame() == current_frame_number:
            return

        print("curr_frame: {}, bf: {}, bff {}".format(current_frame_number, self.detection_list[-1].frame_number, self.detection_list[-2].frame_number))

        # check if two consecutive frames are given
        if self.detection_list[-1].frame_number != current_frame_number - 1:
            return
        if self.detection_list[-2].frame_number != current_frame_number - 2:
            return
        
        # Extrapolation (2 -> 1 -> curr_frame)
        d_1 = self.detection_list[-1]
        d_2 = self.detection_list[-2]

        # get middle points
        mx_1, my_1 = (d_1.x2 - d_1.x1) / 2 + d_1.x1 , (d_1.y2 - d_1.y1) / 2 + d_1.y1
        mx_2, my_2 = (d_2.x2 - d_2.x1) / 2 + d_2.x1 , (d_2.y2 - d_2.y1) / 2 + d_2.y1

        # Drift
        dx = mx_2 - mx_1
        dy = my_2 - my_2 

        #Extrapolation
        xi1 = d_1.x1 + dx
        xi2 = d_1.x2 + dx
        yi1 = d_2.y1 + dy
        yi2 = d_2.y2 + dy

        di = det.Detection(d_1.label, xi1, yi1, xi2 ,yi2, current_frame_number, interpolated = True)
        self.detection_list.append(di)

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
            tube.extrapolate(self.current_frame_number)
            #tube.interpolate(self.current_frame_number)

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
        

    def save(self, path):
        """Save the detected tubes. Recommended after a detection is finished
        
        File Format:
        frame_number, label, x1, y1, x2, y2, interpolated

        file id.tube for each tube

        :path: path to store tube
        :returns: creates a file containing the tubes

        """
        for tube in self.active_tube_list:
            with open(path + "{}.tube".format(tube.id) , 'w+') as f:
                #f.write("Tube:{},{}\n".format(tube.id, len(tube))) 
                for det in tube.detection_list:
                    f.write("{}, {}, {}, {}, {}, {}, {}\n".format(det.frame_number, det.label, det.x1, det.y1, det.x2, det.y2, det.interpolated))

    def load(self, path):
        """
        Loads a TubeGenerator from a collection of Tube files.
        """
        tube_files = [f for f in os.listdir(path) if f.split('.')[-1] == "tube"]

        for fl in tube_files:
            id = fl.split('.')[0]
            with open(path + fl, 'r') as f:
                for i, line in enumerate(f):
                    args = line.strip().split(',')
                    dt = det.Detection(args[1], int(float(args[2])), int(float(args[3])), int(float(args[4])), int(float(args[5])), int(float(args[0])), bool(args[6]))
                    if i == 0:
                        tube = Tube(dt, id)
                    else:
                        tube.detection_list.append(dt)

                self.active_tube_list.append(tube)
