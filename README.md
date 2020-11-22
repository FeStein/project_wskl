# README
This is the project folder associated to the thesis I am writing at
[WSKL](https://www.mv.uni-kl.de/wskl/). At the current state this is a very
early draft status and i will experiment quiet a lot. Don't expect anything to
work properly in this status.

## Installing OpenCV (Anaconda environment)
Assuming you have installed (Mini)conda you can create an Anaconda Environment
and install *FFMPeg* and *OpenCV* with:
```bash
conda create --name opencv
conda activate opencv
conda install -c conda-forge/label/gcc7 ffmpeg
conda install -c menpo opencv
pip install --upgrade pip
pip install opencv-contrib-python
```
`FFMPeg` is needed in case you want to work with videos on *OpenCV* and needs to
be installed before *OpenCV*.





