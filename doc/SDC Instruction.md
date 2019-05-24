# Self Driving Pi Car Setup Log
By: David Tian

Date: 2019-03-25


## Buy Hardware

1. Raspberry Pi 3 Model B+(1.4Ghz, 64 bit Quad core, with Wifi and Bluetooth) $30 [Link](https://amzn.to/2Iki3fb)
1. Pi Case and power supply, $20 (Included in above package)
1. Raspberry Pi Camera Module V2 (5Mp pixel CSI camera), $25 [Link](https://amzn.to/2rKxarh)
1. 64GB MicroSD card.  Better to have 64GB or larger as we will be dealing with a lot of videos and large deep learning model files.

## Set up Raspberry Pi
### Install OS
1. Download [NOOBS](https://www.raspberrypi.org/downloads/noobs/) and unzip onto a formatted micro SD card.
1. Insert SD card into Pi
1. Connect HDMI/mouse/keyboard/2.5A power adapter to Pi board. Make sure to use the 2.5A power adapter, otherwise, it may not boot up, and only show a rainbow colored splash screen with an error stating insufficient power.
1. Pi should boot up.  
1. Select Raspian OS Full Version to install at boot up prompt.  It will take a few minutes to install and use up about 4GB of space.  After installation, you should see a full GUI desktop.  <TODO: image here>
1. First time it boot
	1. OS will ask you to change password for use "pi".  Change it to "rasp"
	1. OS needs to upgrade to latest software.  this may take 10-20 minutes

### Setup Remote Access
Setting up remote access to Pi allows Pi computer to run headless, hence saving us from a monitor and keyboard/mouse. This [video](https://www.youtube.com/watch?v=IDqQIDL3LKg&list=PLQVvvaa0QuDesV8WWHLLXW_avmTzHmJLv&index=3) gives a very good tutorial on how to set up SSH and Remote Desktop.  
1. Get ip address of the Pi 
	- Run `ifconfig`.  Find IP address to be `192.168.1.120` <TODO: image here>
1. Setup SSH and VNC 
	1. Enable SSH Server: 
	- Run `sudo raspi-config` in terminal to start the "Raspberry Pi Software Configuration Tool".  <TODO: image here>
	- For Ras-Config Rev 1.3, Choose `5. Interface Options` -> `SSH` -> `Enable`
	- Connect from Windows via Putty to IP address (192.168.1.120) from 1st step.  Need to type in username/password (pi/rasp)
1. Setup VNC 
	- For Ras-Config Rev 1.3, Choose `5. Interface Options` -> `VNC` -> `Enable`
	- Download RealVNC Viewer from [here](https://www.realvnc.com/en/connect/download/viewer/)
	- Connect to Pi's IP address using Real VNC Viewer.  You will see the same desktop as the one Pi is running.
1. Setup Remote Desktop
	- SSH/VNC needs to be enabled from last 2 steps
	- Run `sudo apt-get install xrdp` to install the Remote desktop Server on Pi
	- Run Remote Desktop from Windows to connect to Pi's address.  You will see a different instance of Pi's desktop.  (So probably better to use VNC, if you want to remote control Pi from your computer)
1. Setup Remote File Access, so we can edit the files on Pi from our own computer.
https://www.juanmtech.com/samba-file-sharing-raspberry-pi/
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install samba samba-common-bin
sudo nano /etc/samba/smb.conf 

# Delete all lines and Paste the following into the file:
[global]
netbios name = Pi
server string = The PiCar File System
workgroup = WORKGROUP

[HOMEPI]
path = /home/pi
comment = No comment
browsable = yes
writable = Yes
create mask = 0777
directory mask = 0777
public = no

# create samba password for user pi.  go ahead and use the same password: rasp
sudo smbpasswd -a pi
New SMB password:
Retype new SMB password:
Added user pi.

# restart samba server
sudo service smbd restart
```
Connect from Windows:  Wait for 30-60 sec and refresh Network, you should see  
\\Raspberrypi\homepi

troubleshoot
sudo service --status-all |grep samba
 [ + ]  samba
 [ - ]  samba-ad-dc
Both service should be [+]
 

1. Remote Debugging from PyCharm <TODO>

### Install USB Camera
1. Plug in the USB Camera

1. Install USB Video Viewer
```
sudo apt-get install cheese
```

1. Launch Video Player.  You should see Live Videos feeds from the USB camera
```
cheese & 
```

	
### Install Pi Camera 
1. Power Pi down by typing this command in a terminal: `sudo shutdown -h now`  (`-h` for shutdown, `-r` for reboot)
1. Connect Pi Camera and enable settings following this [video](https://www.youtube.com/watch?v=T8T6S5eFpqE).
	- Run `sudo raspi-config` 
	- For Ras-Config Rev 1.3, Choose "5. Interface Options" -> "P1 Camera" 
1. Capture Still Image via command line
	1. Run this command: `raspistill -o ~/Desktop/mystill.jpg`. This will bring up the preview video screen, then take a photo and save mystill.jpg on your desktop.  
	1. Double click to open the image to see. <TODO: image here>
	1. `--vflip` option may be useful to flip the image from the camera upside down.
1. Capture Video via command line
	1. Run this command: `raspivid -o ~/Desktop/myvideo.h264 -t 10000`. This will take a 10 sec (10000 ms) video and save myvideo.h264 on your desktop.  
	1. Double click to see the video in default media player.  <image here>
	1. Note there is no sound, since it is just a camera with microphone
1. Capture Still Image via Python by following this [video](https://www.youtube.com/watch?v=qk1IVs5B1GI)
	1. Install picamera python module: 
		```
		sudo apt-get install python-picamera
		```
	1. Initialize picamera: 
		```python
		import picamera
		import time
		camera = picamera.PiCamera()
		```
	1. Capture a still image: 
		```python
		camera.capture('mystill_py.jpg')  
		```
	1. Capture a 5 sec video: 
		```python
		camera.start_recording('myvideo_py.h264')
		time.sleep(5)  # this is in seconds
		camera.stop_recording()
		```

### Setup TensorFlow and OpenCV
TensorFlow is Google's Deep Learning Framework.  OpenCV is an open sourced python computer vision package.  We will use these two to do object detection (stop signs/traffic lights) and lane detection from video feeds.  [Edje Electronics’s video](https://www.youtube.com/watch?v=npZ-8Nj1YwY) and his [github page](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi#4-compile-and-install-protobuf) give an good overview on how to install TensorFlow and how to use it to do object detection from video.

1. Install TensorFlow: (DO NOT use installation from the video above.)  As of March 2019, the most recent version of TensorFlow is version 1.13 (2.0 is still alpha)
	```
	sudo apt-get install libatlas-base-dev
	pip3 install tensorflow 
	```
1. Test TensorFlow Install, by running the "Hello World" program in python3. (Note you MUST run `python3` and NOT `python`, which is python 2.
	```python
import tensorflow as tf
hello = tf.constant('Hello TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
	```
	If you see the following output, then TensorFlow is installed properly.
	```
	Hello TensorFlow!
	```
	Got this warning message. Is this ok? <TODO>
	```
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: builtins.type size changed, may indicate binary incompatibility. Expected 432, got 412	
	```
	
	There are other TF example code that can be found here.
	```
	git clone https://github.com/tensorflow/tensorflow.git
	```
1. Install OpenCV and its dependent packages
	```bash
	sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
	sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
	sudo apt-get install libxvidcore-dev libx264-dev
	sudo apt-get install qt4-dev-tools
	pip3 install opencv-python
	```
1. Install Protobuf, which is needed by TensorFlow's object detection API. It has to be built from source, as no pre-build binaries from [github](https://github.com/protocolbuffers/protobuf/releases) for Pi OS/CPU combination.  The following commands are from [Edje Electronics’s github page](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi#4-compile-and-install-protobuf) 

	```bash
	# First, get the packages needed to compile Protobuf from source
	sudo apt-get install autoconf automake libtool curl

	# Then download the protobuf release from its GitHub repository by issuing. At the time of writing v3.7.0 is the latest version.
	PROTOC_VER=3.7.0
	PROTOC_ZIP=protobuf-all-$PROTOC_VER.tar.gz
	curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VER/$PROTOC_ZIP
	tar -zxvf $PROTOC_ZIP
	cd protobuf-$PROTOC_VER

	# Build the package (this takes about 60 min)
	make
	
	# Check the build (this takes about 2 hours, and may freeze up the Pi system.  Just reboot the pi, and rerun the `make check` again to continue. )
	make check
	
	# Move install to proper system directories
	sudo make install
	
	# Setup protobuf for python: (this takes about 
	cd python
	export LD_LIBRARY_PATH=../src/.libs
	python3 setup.py build --cpp_implementation 
	python3 setup.py test --cpp_implementation
	sudo python3 setup.py install --cpp_implementation
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
	sudo ldconfig
	
	# protoc, the protobuf compiler is compiled and installed.  Try to run it. 
	protoc
	```
	
	You should see the protoc's help page, as below.
	```bash
	pi@raspberrypi:~/Downloads/protobuf-3.7.0/python $ protoc
	Usage: protoc [OPTION] PROTO_FILES
	Parse PROTO_FILES and generate output based on the options given:
	  -IPATH, --proto_path=PATH   Specify the directory in which to search for
								  imports.  May be specified multiple times;
								  directories will be searched in order.  If not
								  given, the current working directory is used.
								  If not found in any of the these directories,
								  the --descriptor_set_in descriptors will be
								  checked for required proto file.
	  --version                   Show version info and exit.
	  -h, --help                  Show this text and exit.
	  [omitted....]	
	
	# Reboot the Pi.  (It is needed for TensorFlow to fully work)
	sudo reboot now
	```
	
	[NOTE] For x86 linux system, then we can just download and unzip directly, as in [this tutorial](https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8). 
	
	```bash
	# Download binary file (not source)
	PROTOC_VER=3.7.0
	PROTOC_ZIP=protoc-$PROTOC_VER-linux-x86_64.zip
	curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VER/$PROTOC_ZIP
	unzip $PROTOC_ZIP -d protoc3
	
	# Move protoc to /usr/local/bin/
	sudo mv protoc3/bin/* /usr/local/bin/
	
	# Move protoc3/include to /usr/local/include/
	sudo mv protoc3/include/* /usr/local/include/
	```

1. Setup TensorFlow Models. Sources are from [TensorFlow github](https://github.com/tensorflow/models.git).  
Note that models under `models/official` are officially maintained by google.  
And models under `models/research` are **NOT** officially maintained by google, but maintained by individual researchers.  

	```bash
	# create a tensorflow model directory
	pi@raspberrypi:~ $ mkdir tensorflow1
	pi@raspberrypi:~ $ cd tensorflow1/
	
	# download model from github (this take about 30 minutes to download 1.5G of model)
	pi@raspberrypi:~/tensorflow1 $ git clone --recurse-submodules https://github.com/tensorflow/models.git
	Cloning into 'models'...
	remote: Enumerating objects: 6, done.
	remote: Counting objects: 100% (6/6), done.
	remote: Compressing objects: 100% (6/6), done.
	Receiving objects:  19% (4920/24902), 101.71 MiB | 854.00 KiB/s  
	[omitted....]

	sudo nano ~/.bashrc
	# add this line to end of  ~/.bashrc so that PYTHONAPTH environment variable includes TF model folders.   
	export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow1/models/research:/home/pi/tensorflow1/research/slim
	
	# close and reopen Terminal to see the PYTHONPATH updated.
	pi@raspberrypi:~ $ echo $PYTHONPATH
	:/home/pi/tensorflow1/models/research:/home/pi/tensorflow1/research/slim
	```

1. Download [Tensorflow Pre Trained Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  Because Pi's limited computing power, we need to choose a model that is fast to run but may not be the most accurate.  
In the [COCO-trained models table](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models), lower number in Speed(ms) column means faster model, and higher number in COCO mAP column means more accurate.  Accounting for both speed and accuracy, these are decent models to use for Pi:
	1. [ssd_mobilenet_v2_coco model](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) seems the best compromise for Pi (time=27ms, accuracy=22).  
	1. [ssd_mobilenet_v1_fpn_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) (time=56ms, accuracy=32) would be another good candidate, if we want more accuracy, but it runs twice as slow as ssd_mobilenet_v2_coco

	```bash
	# We will download ssd_mobilenet_v2_coco model
	pi@raspberrypi:~ $ cd /home/pi/tensorflow1/models/research/object_detection/
	pi@raspberrypi:~/tensorflow1/models/research/object_detection $ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
	--2019-03-26 13:29:05--  http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
	[omitted...]
	2019-03-26 13:30:27 (607 KB/s) - ‘ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz’ saved [51025348/51025348]

	# unzip the tar gz file
	pi@raspberrypi:~/tensorflow1/models/research/object_detection $ tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
	ssdlite_mobilenet_v2_coco_2018_05_09/checkpoint
	ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt.data-00000-of-00001
	ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt.meta
	ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt.index
	ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/saved_model.pb
	ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config
	ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb
	ssdlite_mobilenet_v2_coco_2018_05_09/
	ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/variables/
	ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/
	
	# see the unzipped proto files
	pi@raspberrypi:~/tensorflow1/models/research/object_detection $ ls protos/ssd*
	protos/ssd_anchor_generator.proto  protos/ssd.proto

	# compile the proto files into python wrappers.  need to do it from `research` folders.
	pi@raspberrypi:~/tensorflow1/models/research $ protoc object_detection/protos/*.proto --python_out=.
	
	# after compiling, we should see the python wrapper created from the proto files.
	pi@raspberrypi:~/tensorflow1/models/research $ ls object_detection/protos/ssd*
	object_detection/protos/ssd_anchor_generator_pb2.py
	object_detection/protos/ssd_anchor_generator.proto
	object_detection/protos/ssd_pb2.py
	object_detection/protos/ssd.proto
	```
1. Now all the setup should be done!

### Object Detection
We will continue to follow [Edje Electronics’s video at 14:39](https://www.youtube.com/watch?v=npZ-8Nj1YwY) and his [github page](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi#6-detect-objects) on TensorFlow/OpenCV Object Detection on Raspberry Pi
1. Download his code, [`Object_detection_picamera.py`](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py). 

```bash
# Download the object detection code from Edje's github page
pi@raspberrypi:~/tensorflow1/models/research/object_detection $ wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py

# install other dependent packages needed by TF object detection API
apt-get install libfreetype6-dev
sudo pip3 install pillow jupyter matplotlib cython opencv-python
sudo pip3 install lxml
# there is some erros with lxml install, but doesn't seem to matter.

# Run the object detection python script.
# This will take 1-2 min to load
pi@raspberrypi:~/tensorflow1/models/research/object_detection $ python3 Object_detection_picamera.py
```

### Setup Edge TPU
TensorFlow Processing Unit is Google's specialized hardware to optimized to run deep learning inferences

1. Plug the TPU into USB-C cable, and then plug the USB part into the pi board

1. Power on, and then ssh/vnc into the pi

1. Run the following installation step.  
```bash
wget http://storage.googleapis.com/cloud-iot-edge-pretrained-models/edgetpu_api.tar.gz
tar xzf edgetpu_api.tar.gz
cd python-tflite-source
bash ./install.sh
# output of install.sh
[omitted...]
Using /home/pi/.local/lib/python3.5/site-packages
Finished processing dependencies for edgetpu==1.2.0
```

1. reboot pi to complete the installation.
```bash
sudo reboot
```

1. Test if the installation by trying to identify a parrot image. 
```bash
pi@raspberrypi:~/python-tflite-source/edgetpu $ python3 demo/classify_image.py \> --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
> --label test_data/inat_bird_labels.txt \
> --image test_data/parrot.jpg
```
If you see the following output, then your TPU is connected and working!!
```bash
W0329 23:17:14.486328     814 package_registry.cc:65] Minimum runtime version required by package (5) is lower than expected (10).
---------------------------
Ara macao (Scarlet Macaw)
Score :  0.61328125
---------------------------
Platycercus elegans (Crimson Rosella)
Score :  0.15234375
```

### Running Live Object Detection with TPU
We will be following the classify_capture.py demo (last demo) in [Edge TPU's Demo guide](https://coral.withgoogle.com/tutorials/edgetpu-api/)

```bash
wget -P test_data https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
wget -P test_data/ http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/imagenet_labels.txt
python3 demo/classify_capture_usbcam.py --model test_data/mobilenet_v2_1.0_224_quant_edgetpu.tflite --label test_data/imagenet_labels.txt

python3 demo/object_detection_usbcam.py --model test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --label test_data/coco_labels.txt
```

### Useful Links
- [Edje Electronics’s video](https://www.youtube.com/watch?v=npZ-8Nj1YwY) and his [github page](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi#4-compile-and-install-protobuf) on TensorFlow/OpenCV Object Detection on Raspberry Pi
- [OpenCV with Raspberry Pi Tutorial](https://pythonprogramming.net/raspberry-pi-camera-opencv-face-detection-tutorial/)

### Table of Content
- Hardware List
- RPi Remote Access 
- Software packages installation (Python/OpenCV/numpy/etc)
- TF for CPU
- Object Detection with CPU
- TF for Edge TPU
- Object Detection with CPU
- Object Detection with a single Camera Interface (Pi/USB) a single Object Detection Interface (CPU/TPU)
- Lane Detection
- Stop Sign/Green Light/Red Light Detection (Transfer Learning) https://www.youtube.com/watch?v=Rgpfk6eYxJA
- Distance Sensing
- Steering within Lane
 
 
### Transfer Learning
MobileNet-SSD-V1: fast to run, less accurate
Faster-RCNN-Inception-V2: slow to run, more accurate
FFmpeg split out images from videos

Transfer Learning by Chengwei Zhang  2/11/2019  (HELPFUL)
https://medium.com/swlh/how-to-train-an-object-detection-model-easy-for-free-f388ff3663e
https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb
https://github.com/Tony607/object_detection_demo

Colab Helpful Scripts (Send email and show RAM and GPU memory)
https://colab.research.google.com/drive/1P2AmVHPmDccstO0BiGu2uGAG0vTx444b#scrollTo=VgRfWu26wIBt

Convert model file (proto buffer format) to tflite (flat buffer) format
https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format
https://www.tensorflow.org/lite/convert/cmdline_examples

EdgeTPU (from .tflite to EdgeTPU Model _edgetpu.tflite)
https://coral.withgoogle.com/web-compiler/

# weird TPU error
F0404 10:45:46.228721   20807 usb_driver.cc:834] transfer on tag 1 failed. Abort. generic::unknown: USB transfer error 1 [LibUsbDataOutCallback]
Backend terminated (returncode: -6)
Fatal Python error: Aborted

Thread 0x76faa640 (most recent call first):
  File "/home/pi/python-tflite-source/edgetpu/swig/edgetpu_cpp_wrapper.py", line 110 in RunInference
  File "/home/pi/python-tflite-source/edgetpu/detection/engine.py", line 123 in DetectWithInputTensor
  File "/home/pi/python-tflite-source/edgetpu/detection/engine.py", line 93 in DetectWithImage
  File "/home/pi/python-tflite-source/edgetpu/demo/object_detection_usb.py", line 76 in main

  
# Edge TPU object detection transfer learning tutorial
mkdir D:\David\SelfDrivingCar\EdgeTPU\Docker
cd D:\David\SelfDrivingCar\EdgeTPU\Docker

Download http://storage.googleapis.com/cloud-iot-edge-pretrained-models/docker/obj_det_docker to D:\David\SelfDrivingCar\EdgeTPU\Docker

# this step takes 10-20 min depending on your download speed.
D:\David\SelfDrivingCar\EdgeTPU\Docker>docker build - < obj_det_docker --tag detect-tutorial
Sending build context to Docker daemon  4.608kB
Step 1/13 : FROM tensorflow/tensorflow:1.12.0-rc2-devel
1.12.0-rc2-devel: Pulling from tensorflow/tensorflow
18d680d61657: Pull complete
0addb6fece63: Pull complete
78e58219b215: Pull complete
eb6959a66df2: Pull complete
b612f6150252: Downloading [======================>                            ]  122.2MB/265.8MB
3a3431d93e83: Download complete
def5c38b0d33: Download complete
5838a959ea1d: Download complete
0d228310757c: Download complete
3e8ad7af9b28: Download complete
07710696a7aa: Download complete
8eda15e6480e: Downloading [=====================>                             ]  79.42MB/183.6MB
1204ced585ff: Download complete
31c3d3c34dab: Downloading [=================>                                 ]  19.86MB/57.49MB
94f9c114a883: Waiting


docker run --name detect-tutorial --rm -it --privileged -p 6006:6006  --mount type=bind,src=/d/David/SelfDrivingCar/EdgeTPU/Docker,dst=/tensorflow/models/research/learn_pet detect-tutorial


# ######################################
# Convert PiCar's Server python code to run in python 3 (instead of Python 2)
# Step 1: Back up the picar 3.5 package code
/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg $ p -r picar ~/picar3.5

# Step 2: Do the standard python 2 to python 3 conversion
/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg $ sudo 2to3 -w picar
RefactoringTool: Files that were modified:
RefactoringTool: picar/PCF8591.py
RefactoringTool: picar/__init__.py
RefactoringTool: picar/back_wheels.py
RefactoringTool: picar/filedb.py
RefactoringTool: picar/front_wheels.py
RefactoringTool: picar/SunFounder_PCA9685/PCA9685.py
RefactoringTool: picar/SunFounder_PCA9685/Servo.py
RefactoringTool: picar/SunFounder_TB6612/TB6612.py

# Step 3: copy ball_tracker.py to ball_tracker3.py
pi@raspberrypi:~ $ cd ~/SunFounder_PiCar-V/ball_track/
pi@raspberrypi:~/SunFounder_PiCar-V/ball_track $ cp ball_tracker.py ball_tracker3.py

# Step 4: modify ball_tracker3.py
remove the line `import cv2.cv as cv`
Change `cv.CV_HOUGH_GRADIENT` to `cv2.HOUGH_GRADIENT`
after you are done, the diff should be the following.

pi@raspberrypi:~/SunFounder_PiCar-V/ball_track $ diff ball_tracker.py ball_tracker3.py 
6d5
< import cv2.cv as cv

211c210
<     circles = cv2.HoughCircles(red_hue_image, cv.CV_HOUGH_GRADIENT, 1, 120, 100, 20, 10, 0);
---
>     circles = cv2.HoughCircles(red_hue_image, cv2.HOUGH_GRADIENT, 1, 120, 100, 20, 10, 0);

# Step 5: Run 
pi@raspberrypi:~/SunFounder_PiCar-V/ball_track $ python3 ball_tracker3.py 
DEBUG "back_wheels.py": Set debug off
DEBUG "TB6612.py": Set debug off
DEBUG "TB6612.py": Set debug off
DEBUG "PCA9685.py": Set debug off
DEBUG "front_wheels.py": Set debug off
DEBUG "front_wheels.py": Set wheel debug off
DEBUG "Servo.py": Set debug off
Traceback (most recent call last):
  File "ball_tracker3.py", line 68, in <module>
    bw.speed = 0
  File "/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg/picar/back_wheels.py", line 91, in speed
    self.left_wheel.speed = self._speed
  File "/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg/picar/SunFounder_TB6612/TB6612.py", line 62, in speed
    self._pwm(self._speed)
  File "/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg/picar/back_wheels.py", line 46, in _set_a_pwm
    self.pwm.write(self.PWM_A, 0, pulse_wide)
  File "/usr/local/lib/python3.5/dist-packages/SunFounder_PiCar-1.0.1-py3.5.egg/picar/SunFounder_PCA9685/PCA9685.py", line 229, in write
    self._write_byte_data(self._LED0_OFF_L+4*channel, off & 0xFF)
TypeError: unsupported operand type(s) for &: 'float' and 'int'

# python 2.7 package location
/usr/local/lib/python2.7/dist-packages/SunFounder_PiCar-1.0.1-py2.7.egg/picar

# Run wheel test (use python for v2 and python3 for v3)
import picar
picar.back_wheels.test()
picar.front_wheels.test()

pi@raspberrypi:~/py3/SunFounder_PiCar-V/remote_control $ sudo ./start
Server running
Traceback (most recent call last):
  File "manage.py", line 8, in <module>
    from django.core.management import execute_from_command_line
ImportError: No module named 'django'

pip3 install Django

issue: 
    def map(self, x, in_min, in_max, out_min, out_max):
        '''To map the value from arange to another'''
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# How to train an object detection model with Tensorflow?
Edje Electronics: 
Label data with Label Img Tool
- Download LabelImg from https://github.com/tzutalin/labelImg.  Windows binaries is prebuilt.  Mac and Linux can be build from source.
- Click Open Dir to point to training image folders
- Turn on Auto Save xml: View Menu -> Auto Saving
- Useful shortcuts: W to create rectangular box, D to go to next file, A to go to previous file
- Will see an xml along with each picture file.  xml files specify the bounding box and type of object labeled by you

<annotation>
	<folder>train</folder>
	<filename>car_video_5_145.png</filename>
	<path>D:\temp\train\car_video_5_145.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>320</width>
		<height>240</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>Red Traffic Light</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>260</xmin>
			<ymin>81</ymin>
			<xmax>273</xmax>
			<ymax>108</ymax>
		</bndbox>
	</object>
	<object>
		<name>Person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>192</xmin>
			<ymin>105</ymin>
			<xmax>208</xmax>
			<ymax>137</ymax>
		</bndbox>
	</object>
</annotation>

Transfer learning from ssd_model
https://www.youtube.com/watch?v=Rgpfk6eYxJA
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10   
model zoo: faster rcnn model for coco