import cv2
import numpy as np

def calculate_IOU(det1, det2):
    """Calculates the intersection over union between two detections

    :det1: Frist detction
    :det2: Second detection
    :returns: IoU value {0..1}

    """
    pass

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


    def update(self, detection_list):
        """Updates the current tubes by the detections given. Will initialize
        new tubes when necessary and close unnecessary tubes when not needed.

        :detections: List of detections (output of the detection per frame)

        """
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
    print("no errors")
