import cv2
import numpy as np

def calculate_IOU(det1, det2):
    """Calculates the intersection over union between two detections

    :det1: Frist detction
    :det2: Second detection
    ---------------------------------------------------------------------------
    :returns: IoU value {0..1}

    """
    # compute area of both bounding boxes
    box1Area = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
    box2Area = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)

    # compute corner points of intersection
    x_left = max(det1.x1, det2.x1)
    x_right = min(det1.x2, det2.x2)
    y_bottom = max(det1.y1, det2.y1)
    y_top = min(det1.y2, det2.y2)
    

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # compute area of intersection/overlap
    interArea = (x_right - x_left) * (y_top - y_bottom)

    iou = interArea / (box1Area + box2Area - interArea)

    print(interArea, box1Area, box2Area)
    
    if iou < 0.0 or iou > 1.0:
        raise ValueError("IoU calculation went wrong (out of bounds)")

    return iou

class Tube():

    """Tube class, which represents an object present on consecutive frames"""

    def __init__(self, detection):
        """Initializes a new tube """
        pass

    def add(self, detection):
        """Adds a new detection in a new frame to the tube

        :detection: TODO
        :returns: TODO

        """
        pass

class TubeGenerator():

    """Generator class to keep track of all tubes in a video sequence"""

    def __init__(self):
        """Initializes a new tube generator. Recommended use: initalize it
        before you loop over the video frames """

        self.current_frame_number = 0


    def update(self, detection_list):
        """Updates the current tubes by the detections given. Will initialize
        new tubes when necessary and close unnecessary tubes when not needed.

        :detections: List of detections (output of the detection per frame)

        """
        self.current_frame_number += 1
        pass

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
    d1 =  detection.Detection('car', 507, 205, 561, 241)
    d2 =  detection.Detection('traffic light', 139, 105, 151, 121)
    iou = calculate_IOU(d1,d2) 
    print(iou)
    print("no errors")
