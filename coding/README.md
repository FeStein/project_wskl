# coding
This folder contains some temporary coding scripts and snippets I'm trying to
implement. Mostly not working, otherwise it would already be in the *src/*
folder.

## learning/

In this folder, i will keep some (simple) scripts I write in order to learn the
tools I need.

### cascade_detection

some detection examples using haar cascade classifiers (just basic stuff)

#### detect_face.py

Detecting the face and parts the face from video streams (using a sample video
stream of myself). Following along with 
[Sentdex Tutorial](https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/).
Stuff is based on haar feature-based Cascade Classifiers. Theory is given in the OpenCV
[Documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).

### yolo

Experiments using the YOLOL algorithm and the FOSS implementation *darknet*. 

#### detection_in_video.py

This kinda combines
[this](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
tutorial with classifying in videos. As input video stream a SmartFactory
Youtube Video is used. Currently working add proper arg-parse and logging. To
run the script properly use:
```bash
python3 detection_in_video.py -vs ../../../data/yt_wskl/youtube_wskl_1.mp4
```
You can specify additional conditions with:
|  flag      |    behavior       |  default   	|
| ------------- |-------------  | ------- |
|    `-c`   |    confidence          | 0.5       |
|    `-t`   |    threshold          | 0.3       |
|    `-l`   |    logging          | False       |
The `labels.txt` file is an overview over all labels on which the model is
trained on. Currently a trained model form darknet is used, which is trained on
80 different keywords using the COCO dataset. The back-end algorithm is YOLOv3.


