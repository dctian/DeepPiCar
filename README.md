#### [DeepPiCar Series](https://towardsdatascience.com/tagged/deep-pi-car)

# DeepPiCar — Part 1: How to Build a Deep Learning, Self Driving Robotic Car on a Shoestring Budget

## An overview of how to build a Raspberry Pi and TensorFlow powered, self-driving robotic car

### By [David Tian](https://towardsdatascience.com/@dctian)

Hacker, tinkerer, and engineer. I am passionate about machine learning, AI, and
anything technology related.

![](https://cdn-images-1.medium.com/max/800/1*4GhtKM-eyuYqEpZnnUJZ9w@2x.jpeg)

### Introduction

Today, Tesla, Google, Uber, and GM are all trying to create their own
self-driving cars that can run on real-world roads. Many analysts predict that
within the next 5 years, we will start to have fully autonomous cars running in
our cities, and within 30 years, nearly ALL cars will be fully autonomous.
Wouldn’t it be cool to build your very own self-driving car using some of the
same techniques the big guys use? In this and next few articles, I will guide
you through how to build your own physical, deep-learning, self-driving robotic
car from scratch. You will be able to make your car detect and follow lanes,
recognize and respond to traffic signs and people on the road in under a week.
Here is a sneak peek at your final product.

![](https://cdn-images-1.medium.com/max/600/1*3sMJxWJ34vQH0WobdFPVAA.jpeg)

Lane Following

![](https://cdn-images-1.medium.com/max/600/1*bYqrTsiMnoaKu9CfjewlEg.jpeg)

Traffic Sign and People Detection (right) from
DeepPiCar’s DashCam</span>

### Our Road Map

*Part 2*: I will list what hardware to buy and how to set them up. In short, you
will need a [Raspberry
Pi](https://www.amazon.com/CanaKit-Raspberry-Power-Supply-Listed/dp/B07BC6WH7V/)
board($50), [SunFounder PiCar
kit](https://www.amazon.com/SunFounder-Raspberry-Graphical-Programming-Electronic/dp/B06XWSVLL8)
($115), [Google’s Edge TPU](https://coral.withgoogle.com/products/accelerator)
($75) plus a few accessories, and how each part is important in later articles.
The total cost of the materials is around $250–300. We will also install all the
software drivers needed by Raspberry Pi and PiCar.

![](https://cdn-images-1.medium.com/max/400/1*H7mwt6TcJtZc28fsKh42xg.jpeg)

Raspberry Pi 3 B+

![](https://cdn-images-1.medium.com/max/400/1*LUD3NFk4hCz5wFpRWSGODQ.jpeg)

SunFounder PiCar-V Robotic Car Kit

<img src="https://cdn-images-1.medium.com/max/600/1*RIddRse2MoaJtSFes6VkgQ.jpeg" width="300" />

Google Edge TPU Accelerator

*Part 3*: We will set up all the Computer Vision and Deep Learning software
needed. The main software tools we use are [Python](https://www.python.org/)
(the de-facto programming language for Machine Learning/AI tasks), [OpenCV
](https://github.com/opencv/opencv)(a powerful computer vision package) and
[Tensorflow ](https://www.tensorflow.org/)(Google’s popular deep learning
framework). Note all the software we use here are FREE and open source!

![](https://cdn-images-1.medium.com/max/800/1*_jABdMfUVcyPdi5b3zlfVg.jpeg)

*Part 4*: With the (tedious) hardware and software setup out of the way, we will
dive right into the FUN parts! Our first project is to use python and OpenCV to
teach DeepPiCar to navigate autonomously on a winding single lane road by
detecting lane lines and steer accordingly.

![](https://cdn-images-1.medium.com/max/800/1*cVqpqZ129JiiQZxZwqMlMg.jpeg)
<span class="figcaption_hack">Step-by-Step Lane Detection</span>

*Part 5*: we will train DeepPiCar to navigate the lane autonomously without having
to explicitly write logic to control it, as was done in our first project. This
is achieved by using “behavior cloning”, where we use just the videos of the
road and the correct steering angles for each video frame to train DeepPiCar to
drive itself. The implementation is inspired by [NVIDIA’s
DAVE-2](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
full-sized autonomous car, which uses a deep Convolutional Neural Network to
detect road features and make the correct steering decisions.

<span class="figcaption_hack">Lane Following in Action</span>

Lastly, in *Part 6*: We will use deep learning techniques such as [single shot
multi-box object detection](https://arxiv.org/abs/1512.02325) and [transfer
learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
to teach DeepPiCar to detect various (miniature) traffic signs and pedestrians
on the road. And then we will teach it to stop at red lights and stop signs, go
on green lights, stop to wait for a pedestrian to cross, and change its speed
limit according to the posted speed signs, etc.

![](https://cdn-images-1.medium.com/max/1200/1*Hw7r95umdwnzK2EPTayvfg.jpeg)
<span class="figcaption_hack">Traffic Signs and People Detection Model Training in TensorFlow</span>

### Prerequisite

Here are the prerequisites of these articles:

* First and foremost is the willingness to *tinker and break things*. Unlike in a
car simulator, where everything is deterministic and perfectly repeatable,
real-world model cars can be unpredictable and you must be willing to get your
hands dirty and start to tinker with both the hardware and software.
* Basic *Python programming* skills. I will assume you know how to read python
code and write functions, if statements and loops in python. Most of my code is
well documented, specifically the harder to understand parts.
* Basic *Linux operating system* knowledge. I will assume you know how to run
commands in Bash shell in Linux, which is Raspberry Pi’s operating system. My
articles will tell you exactly which commands to run, why we run them, and what
to expect as output.
* Lastly, you will need about *$250-$300* to buy all the hardware and working PC
(Windows/Mac or Linux). Again, all the software used will be free.


### What’s Next

That’s all for the first article. I will see you in Part 2 where we will get our
hands dirty and build a robotic car together!

Here are the links to the whole guide:

Part 1: [Overview](https://medium.com/@dctian/deeppicar-part-1-102e03c83f2c?source=your_stories_page---------------------------)
(This article)

Part 2: [Raspberry Pi Setup and PiCar
Assembly](https://medium.com/@dctian/deeppicar-part-2-8512be2133f3?source=your_stories_page---------------------------)

Part 3: [Make PiCar See and
Think](https://medium.com/@dctian/deeppicar-part-3-d648b76fc0be?source=your_stories_page---------------------------)

Part 4: [Autonomous Lane Navigation via
OpenCV](https://medium.com/@dctian/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96?source=your_stories_page---------------------------)

Part 5:
[Autonomous](https://medium.com/@dctian/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96?source=your_stories_page---------------------------)[
Lane Navigation via Deep
Learning](https://medium.com/@dctian/deeppicar-part-5-lane-following-via-deep-learning-d93acdce6110?source=your_stories_page---------------------------)

Part 6: [Traffic Sign and Pedestrian Detection and
Handling](https://medium.com/@dctian/deeppicar-part-6-963334b2abe0?source=your_stories_page---------------------------)




