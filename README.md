# README
This is the project folder associated to the thesis I am writing at
[WSKL](https://www.mv.uni-kl.de/wskl/). At the current state this is a very
early draft status and i will experiment quiet a lot. Don't expect anything to
work properly in this status.

Quick note: The instructions are mostly for myself and people I am working with.
Therefore they are tailored towards my own needs and one should not expect that
they work for him. I mainly develop on an Ubuntu 20.04 Machine and use a MacBook
Pro to remotely connect to it. Your setup and installation process could differ,
so please use the install instructions with care and make sure you understand
what you are doing before you apply anything blindfolded (Check that the paths
are correct for example).

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

### Installing OpenCV from Source (core modules)

Configure your `CMakeListings.txt` before running cmake (enable and specify
python3).

```bash
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libwebp-dev
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
sudo apt-get install libgtk2.0-dev
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
mkdir -p opencv-build && cd opencv-build
cmake  ../opencv-master
sudo make install -j4
```
You should now run `import cv2` and test if it runs without producing errors. 

### Installing Darknet with OpenCV support

After installing OpenCV we can try to install darknet to implement the YOLO
Algorithm:
```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet && make -j4
```
If you want to compile using OpenCV, open the Makefile and set `OPENCV=1`. Same
applies to CUDA but i didn't tried that out yet.I added a system variable by
adding the Darknet path to my system configuration:
```bash
export DARKNET_PATH=/home/felix/Programs/darknet
```
