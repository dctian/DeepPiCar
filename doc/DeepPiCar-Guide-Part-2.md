### DeepPiCar: How to Build a Deep Learning, Self Driving Robotic Car on a
Shoestring Budget — Part 1

### Hardware Setup Guide

Welcome back! In this guide, we will first go over what hardware to purchase and
why we need them. Next, we will set them up and get ready for the Software
Installation Guide (Part 2)

### Hardware Supply List

![](https://cdn-images-1.medium.com/max/1000/1*BxFrnHdOkpuFjkZDopxi1g.jpeg)

* 1 x [Raspberry Pi 3 Model B+ kit with 2.5A Power
Supply](https://www.amazon.com/CanaKit-Raspberry-Power-Supply-Listed/dp/B07BC6WH7V/)
($50) This is the brian of your DeepPiCar. This latest model of Raspberry Pi
features a 1.4Ghz 64-bit Quad-Core processor, dual band wifi, Bluetooth, 4 USB
ports, and an HDMI port. I recommend this kit (over just the Raspberry Pi board)
because it comes with a power adapter, which you need to plug in while doing
your non-driving coding and testing, and two chip heat sinks, which will prevent
your Raspberry Pi CPU from overheating.
* 1 x [64 GB micro SD
Card](https://www.amazon.com/Kingston-64GB-microSDHC-microSD-SDCS/dp/B079GVC5B8/)
($8) This is where your Raspberry Pi’s operating system and all of our software
will be stored. Any brand of micro SD card should work fine. You may just have
one lying around your house. 32GB should be fine as well. I choose 64 GB because
I like to record lots of videos of while my car is driving so that I can analyze
its behavior later.

![](https://cdn-images-1.medium.com/max/1000/1*Vadg2dAN-nb1CPO9Kp5X-A.jpeg)

* 1 x [SunFounder PiCar-V
kit](https://www.amazon.com/SunFounder-Raspberry-Graphical-Programming-Electronic/dp/B06XWSVLL8/)
($115) This is the main body of DeepPiCar. Make sure you get the Model V as
shown above(a.k.a. Version 2.0). It comes with everything you need in a running
car, except for the Raspberry Pi and the batteries. There are a number of
Raspberry Pi car kits on the market, I chose this car kit because it comes with
an open source python API to control the car, whereas other vendors have its
proprietary API or C based API. As we know, python is now the language of choice
for machine learning and deep learning. Also, open source is important as we may
tinker with the internals of the car API ourselves if we find bugs in the API
without having to wait for the manufacturer to provide software updates.
* 4 x [18650 batteries and 1 x battery charger
](https://www.amazon.com/Garberiel-5000mAh-Rechargeable-Flashlight-Universal/dp/B07QD2432Q/)($20)
You may get any 18650 batteries and compatible charger. These batteries are for
high drain applications, such as driving a Raspberry Pi and the PiCar. PiCar
takes only two batteries, but you always want to have another freshly charged
pair around, so that you can keep your car running on the tracks at all times. I
recommend charging both sets at night, so you won’t have to worry about dead
batteries during testing.

![](https://cdn-images-1.medium.com/max/1000/1*gmoi99Uadi8TyD5axK02NA.jpeg)

* 1 x [Google Edge TPU USB
Accelerator](https://coral.withgoogle.com/products/accelerator) ($75) Every hero
needs a sidekick. Google’s Edge TPU (Edge means it's for mobile and embedded
devices and TPU stands Tensor Processing Unit) is a wonderful add on to the
Raspberry Pi board. While the Pi CPU packs a lot of computing power in a tiny
bundle, it is NOT designed to do deep learning. Google’s newly released Edge
TPU(March 2019), on the other hand, is specifically designed to run deep
learning models written in TensorFlow. In Part 4 of this series, we will build a
real-time traffic sign detection model in TensorFlow. This model is 200+ layers
deep! Running this model on Raspberry Pi’s CPU alone can only process 1 Frame
per Second (FPS) which is hardly real-time. Plus it consumes 100% of the CPU and
makes all the other programs non-responsive. But with the help of Edge TPU, we
can now process 12 FPS, which is adequate for real-time work. And our CPU stays
cool and can be utilized to do other processing tasks, like controlling the car.

![](https://cdn-images-1.medium.com/max/500/1*79NOQOk2Pd3m0EudNPOMGA.jpeg)

![](https://cdn-images-1.medium.com/max/1000/1*v3Bx16IahflxijtPTcWv7g.jpeg)

* 1 x [Set of Miniature Traffic
Signs](https://www.amazon.com/Wooden-Street-Playset-Traffic-Perfect/dp/B01A8XTHHA/)
and a few Lego figurines ($15) You may not need to buy them if your younger ones
have some of these toy signs and Lego figurines in the playroom. You can use
whatever signs you find to train the model, just make sure they are *not TOO
BIG*!

![](https://cdn-images-1.medium.com/max/1000/1*wxu8VHnjNBL5scfsIqv1uQ.jpeg)

* (Optional) 1 x [170 degree Wide Angle USB
Camera](https://www.amazon.com/ELP-2-1mm-Camera-industrial-system/dp/B01N07O9CQ)
($40). This is an optional accessory. I bought it to replace the stock camera
that came with the SunFounder PiCar so that the car can have a wide field of
vision. The stock camera is great, but not as wide angle as I like, and can’t
see lane lines that are 3–4 inches in front of the front wheels. I wrote the
lane following code in Part 3 with the stock camera initially. After trying a
few lens, I found that the lane following stability greatly increased with this
wide angle camera. It is nice to have control of both your hardware and
software, because you may resort to a hardware solution if a problem can’t be
easily solved via software alone. 
* USB Keyboard/Mouse and Monitor that takes HDMI input. You only need these during
the initial setup stage of the Pi. Afterward, we can remote control the Pi via
VNC or Putty. 
* A desktop or laptop computer running Windows/Mac or Linux, which I will refer to
as “PC” here onwards. We will use this PC to remote access and deploy code to
the Pi computer. 

Sometimes, it surprises me that Raspberry Pi, the brain of our car is only about
$30 and cheaper than many of our other accessories. Indeed, the hardware is
getting cheaper and more powerful over time, and software is completely free and
abundant. Don’t we live in a GREAT era?!

This is the end product when the assembly is done. I used a wide-angle camera
here.

![](https://cdn-images-1.medium.com/max/1000/1*ZczTjWEZuBf1Plhpv--x2Q.jpeg)

<br> 

### Hardware Set up

#### Raspberry Pi Operating System Set up (1 Hour)

* Follow this excellent [step-by-step
guide](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) to
install the NOOBS Raspbian Operating System (a variate of Linux) onto a micro SD
card. It would take about 20 min and about 4GB of disk space. After installation
and reboot, you should see a full GUI desktop like below. This feels like you
are in a Windows or Mac GUI environment, doesn’t it?

![](https://cdn-images-1.medium.com/max/1000/1*0IIdVI_gbhYuY-m4D-aMpA.jpeg)

* During installation, Pi will ask you to change the password for the default user
`pi`. Let’s set the password to `rasp`, for example.
* After the initial installation, Pi may need to upgrade to the latest software.
This may take another 10–15 minutes.

#### Setup Remote Access

Setting up remote access to Pi allows Pi computer to run headless saves us from
having to connect a monitor and keyboard/mouse to it all the time. This
[video](https://www.youtube.com/watch?v=IDqQIDL3LKg&list=PLQVvvaa0QuDesV8WWHLLXW_avmTzHmJLv&index=3)
gives a very good tutorial on how to set up SSH and VNC Remote Access. Here are
the steps.

* Open the Terminal application, as shown below. Terminal is a very important
program, as most of our command in later articles will be entered from Terminal.

![](https://cdn-images-1.medium.com/max/1000/1*oWEdMfJTH1TuRIN2JQfeng.jpeg)

* Find the IP address of the Pi by running `ifconfig`. In this case, my Pi’s IP
address is `192.168.1.120`. 

    pi@raspberrypi:~ $ 
    wlan0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 
      netmask 255.255.255.0  broadcast 192.168.1.255

* Run `sudo raspi-config` in Terminal to start the “Raspberry Pi Software
Configuration Tool”. You may be prompted to type in the password for user `pi`

![](https://cdn-images-1.medium.com/max/1000/1*PG1uTJWXdeTUOaSYn9I11A.jpeg)

* Enable SSH Server: Choose `5. Interface Options` -> `SSH` -> `Enable`
* Enable VNC Server: Choose `5. Interface Options` -> `VNC` -> `Enable`
* Download and [install RealVNC
Viewer](https://www.realvnc.com/en/connect/download/viewer/) onto your PC.
* Connect to Pi’s IP address using Real VNC Viewer. You will see the same desktop
as the one Pi is running.
* At this point, you can safely disconnect the monitor/keyboard/mouse from the Pi
computer, leaving just the power adapter plugged in.

![](https://cdn-images-1.medium.com/max/1000/1*E-Z-fOghfDqd6u8jJGuRfg.jpeg)

#### Install USB Camera

The device driver for the USB camera should already come with Raspian OS. We
will install a Video Camera Viewer so we can see live videos.

* Take the USB Camera out of PiCar kit and plug into Pi computer’s USB port
* Run `sudo apt-get install cheese` from the terminal to install “Cheese”, the
camera viewer.

    pi@raspberrypi:~ $ 
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    ....
    cheese is the newest version (3.22.1-1).

* Launch Cheese app by `Raspberry Pi button(Top Left Corner)`-> `Sound & Video` ->
`Cheese` You should see a live video feed displayed like the picture above.

#### SunFounder PiCar-V Software Configuration (Deviations from the manual)

Before assemblying PiCar, we need to download a patched version of PiCar python
API code. SunFounder release a server version and client version of its Python
API. The Client API code runs on your PC, and it uses Python version 3. However,
the Server API code runs on the Pi computer/PiCar, and it uses Python version 2,
which is an outdated version. Since the self-driving programs that we will write
will exclusively run on the Pi Computer/PiCar, the PiCar Server API to run in
Python 3 also. Fortunately, all of SunFounder’s API code are open source on
[Github](https://github.com/sunfounder/SunFounder_PiCar-V), I make a [fork
](https://github.com/dctian/SunFounder_PiCar-V)and updated the entire repo to
Python 3. (I will submit my changes to SunFounder soon, so it can be merged back
to the main repo, hopefully.)

For the time being, run the following commands instead of the software commands
in the manual. You shouldn’t have to run commands on Pages 20–26 of the manual.

    # route all calls to python (version 2) to python3, 
    # pip (version 2) to pip3, even in sudo mode 
    # note: `sudo abcd` runs `abcd` command in administrator mode
    alias python=python3
    alias pip=pip3
    alias sudo='sudo '

    # Download patched PiCar-V driver API, and run its set up
    cd ~
    git clone 
    cd ~/SunFounder_PiCar/picar
    git clone 
    cd ~/SunFounder_PiCar
    sudo python setup.py install

    # Download patched PiCar-V applications
    # and install depedent software
    cd ~
    git clone 
    cd SunFounder_PiCar-V
    sudo ./install_dependencies

Now all required hardware drivers are installed. 

#### SunFounder PiCar-V Assembly

The assembly process closely reassembles building a complex Lego set, and the
whole process takes about 2 hours, a lot of hand-eye coordination and is loads
of fun. (You may even involve your younger ones during the construction phase.)
PiCar Kit comes with a printed step-by-step instructional manual. But I
recommend these two *additional *resources.

* [PDF version of the instructional
manual](https://www.sunfounder.com/learn/download/X1BWQ19SYXNwYmVycnlfUGlfU21hcnRfVmlkZW9fQ2FyX1YyLjAucGRm/dispi).
The print manual is small, and diagrams may not be printed very clearly, whereas
the PDF version is crystal clear, can be searched and zoomed in for more
details. I found it very helpful with the PDF on my laptop during the assembly
phase.
* [YouTube 4-part instructional
videos](https://www.youtube.com/watch?v=Tg_g4YoAZdc&list=PLwWF-ICTWmB6TJ9_kBLL4r_P4yszQycoU)
published by SunFounder. Unfortunately, these videos are for an older version of
PiCar, so some parts (like the servo motor assembly) are different. But most
parts and assembling techniques are the same. So if you are scratching your head
at a particular diagram in the assembly manual, you may want to take a look at
the relevant parts of the videos. I wish SunFounder would publish a new set of
videos for the new PiCar-V kit.

<span class="figcaption_hack">Assembly Videos (4 Parts) for an Older Version of PiCar</span>

#### <br> 

#### When the Rubber Hits the Road!

Now that all the basic hardware and software for the PiCar is in place, let’s
try to run it!

* Connect to PiCar via VNC from PC
* Make sure batteries are in, toggle the switch to ON position and unplug the
micro USB charging cable. Note that your VNC remote session should still be
alive. 
* In a Pi Terminal, run the following commands. You should 1) see the car going
faster, and slower when you issue `picar.back_wheel.test()`, and 2) see the
front wheels steer left, center and right when you
issue`picar.front_wheel.test()`. To stop these tests, press Ctrl-C. To exit the
python program, press Ctrl-D.

    pi@raspberrypi:~/SunFounder_PiCar/picar $ 
     
    Python 3.5.3 (default, Sep 27 2018, 17:25:39) 
    [GCC 6.3.0 20170516] on linux
    Type "help", "copyright", "credits" or "license" for more information.

    >>> 
    >>> 

    >>> 
    DEBUG "front_wheels.py": Set debug off
    DEBUG "front_wheels.py": Set wheel debug off
    DEBUG "Servo.py": Set debug off
    turn_left
    turn_straight
    turn_right

    >>> 
    DEBUG "back_wheels.py": Set debug off
    DEBUG "TB6612.py": Set debug off
    DEBUG "TB6612.py": Set debug off
    DEBUG "PCA9685.py": Set debug off
    Forward, speed = 0
    Forward, speed = 1
    Forward, speed = 2
    Forward, speed = 3
    Forward, speed = 4
    Forward, speed = 5
    Forward, speed = 6
    Forward, speed = 7
    Forward, speed = 8
    Forward, speed = 9
    Forward, speed = 10
    Forward, speed = 11

* If you run into errors or don’t see the wheels moving, either something is wrong
with your hardware connection or software set up. For the former, please double
check your wires connections, and make sure the batteries are fully charged. For
the latter, please post a message with detailed steps you followed, an error
message in the comment section down below, and I will try to help. 

### What’s Next

Congratulations, you now have a PiCar that can see via Cheese, and its speed and
steering can be controlled via python 3 code! It is not quite a Deep Learning
Car yet, but we are well on our way to that. Whenever you are ready, please give
me some claps and head on over to Part 2, the installation guide for computer
vision and deep learning.
