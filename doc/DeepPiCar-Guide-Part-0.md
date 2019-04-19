### DeepPiCar: How to Build a Deep Learning, Self Driving Robotic Car on a
Shoestring Budget — Part 0

![](https://cdn-images-1.medium.com/max/1000/1*4GhtKM-eyuYqEpZnnUJZ9w@2x.jpeg)

### Introduction

Today, Telsa, Google, Uber, and GM are all trying to create their own
self-driving cars that can run on real-world roads. Many analysts predict that
within the next 5 years, we will start to have fully autonomous cars running in
our cities, and within 30 years, nearly ALL cars will be fully autonomous.
Wouldn’t it be cool to build your very own self-driving car using some of the
same techniques the big guys use? In this and next few articles, I will guide
you through how to build your own physical, deep-learning, self-driving robotic
car from scratch. You will be able to make your car detect and follow lanes,
recognize and respond to traffic signs and people on the road. Here is a sneak
peek at your final product.

![](https://cdn-images-1.medium.com/max/750/1*3sMJxWJ34vQH0WobdFPVAA.jpeg)

![](https://cdn-images-1.medium.com/max/750/1*bYqrTsiMnoaKu9CfjewlEg.jpeg)
<span class="figcaption_hack">Lane Following (left) and Traffic Sign and People Detection (right) from
DeepPiCar’s DashCam</span>

### Our Road Map

Part 1: I will list what hardware to buy and how to set them up. In short, you
will need a [Raspberry
Pi](https://www.amazon.com/CanaKit-Raspberry-Power-Supply-Listed/dp/B07BC6WH7V/)
board($50), [SunFounder PiCar
kit](https://www.amazon.com/SunFounder-Raspberry-Graphical-Programming-Electronic/dp/B06XWSVLL8)
($115), [Google’s Edge TPU](https://coral.withgoogle.com/products/accelerator)
($75) plus a few accessories, and how each part is important in later articles.
The total cost of the materials is around $250–300.

![](https://cdn-images-1.medium.com/max/500/1*H7mwt6TcJtZc28fsKh42xg.jpeg)

![](https://cdn-images-1.medium.com/max/500/1*LUD3NFk4hCz5wFpRWSGODQ.jpeg)

![](https://cdn-images-1.medium.com/max/750/1*RIddRse2MoaJtSFes6VkgQ.jpeg)
<span class="figcaption_hack">Raspberry Pi 3 B+ (left), SunFounder PiCar-V (middle), Google Edge TPU (right)</span>

Part 2: We will set up all the software needed for later stages. The main
software tools we use are [Python](https://www.python.org/) (the de-facto
programming language for Machine Learning/AI tasks), [PiCar’s Python
API](https://github.com/sunfounder/SunFounder_PiCar-V) (to control the steering
and speed), [OpenCV ](https://github.com/opencv/opencv)(a powerful computer
vision package) and [Tensorflow ](https://www.tensorflow.org/)(Google’s popular
deep learning framework). Note all the software we use here are FREE and open
source!

<br> 

![](https://cdn-images-1.medium.com/max/500/1*_wbDOXgPIxEsHLu7KEFY_w.png)

![](https://cdn-images-1.medium.com/max/750/1*f5ySeQqn5E8SpRBZXKcvCQ.jpeg)

![](https://cdn-images-1.medium.com/max/500/1*FseKzrJydt1A8eV7pY0FDQ.png)

Part 3: With the (tedious) hardware and software setup out of the way, we will
dive right into the FUN parts! Our first project is to use python and OpenCV to
teach DeepPiCar to navigate autonomously on a winding single lane road by
detecting lane lines and steer accordingly.

![](https://cdn-images-1.medium.com/max/1000/1*cVqpqZ129JiiQZxZwqMlMg.jpeg)
<span class="figcaption_hack">Step-by-Step Lane Detection</span>

Part 4: Our second project will be using deep learning techniques such as
[single shot multi-box object detection](https://arxiv.org/abs/1512.02325) and
[transfer
learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
to teach DeepPiCar to detect various (miniature) traffic signs and people on the
road. And then we will teach it to stop at red lights and stop signs, go on
green lights, stop to wait for a person to cross, and change its speed limit
according to the posted speed signs, etc.

![](https://cdn-images-1.medium.com/max/1500/1*Hw7r95umdwnzK2EPTayvfg.jpeg)
<span class="figcaption_hack">Traffic Signs and People Detection Model Training in TensorFlow</span>

Lastly, we will train DeepPiCar to navigate the lane autonomously without having
to explicitly write logic to control it, as was done in our first project. This
is achieved by using “behavior cloning”, where we use just the videos of the
road and the correct steering angles for each video frame to train DeepPiCar to
drive itself. The implementation is inspired by [NVIDIA’s
DAVE-2](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
full-sized autonomous car, which uses a Convolutional Neural Network to detect
road features and make the correct steering decisions.

Lane Following in Action

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
* Lastly, you will need about *$250-$300* to buy all the hardware. Again, all the
software will be free.

### Further Thoughts [Optional]

This is optional reading, as I try to cover everything you need to know in my
articles. However, if you want to dive deeper into deep learning, (pun
intended), in additional to the links I provided throughout the article, here
are some more resources to check out.

[Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng)’s Machine Learning and Deep
Learning courses on [Coursera](https://www.coursera.org). It was these courses
that got me super excited about Machine Learning and AI, and gave me the
inspiration to create DeepPiCar.

* [Machine Learning](https://www.coursera.org/learn/machine-learning) (FREE): This
course covers traditional Machine learning techniques, such as Linear
regression, Logistic regression, and Support Vector Machines, etc, as well as
Neural Networks. It was created back in 2012, so some of the tools it uses,
namely Matlab/Octave, are out of fashion, and it didn’t talk about deep learning
in great length. But the concepts it teaches you are invaluable. You only need
high school level math and some basic programming skills to take the course and
Dr. Ng explains difficult concepts like backpropagation extremely well. It takes
about 3 months to complete this course.

![](https://cdn-images-1.medium.com/max/1000/1*V71ojAC3PXiJfK30Xr9mjA.jpeg)

* [Deep Learning 5-Course
Specialization](https://www.deeplearning.ai/deep-learning-specialization/) (FREE
or $50/month if you want to get the certificate): This course was introduced in
early 2018. So it covers all the latest AI research up to that time, such as
Fully Connected Neural Networks, Convolutional Neural Network (CNN), and
Sequence Models (RNN/LSTM). This course was such a treat for me. As an engineer,
I always wonder how some of the cool gadgets work, such as how does Siri respond
to your questions, and how does a car recognize objects on the road, etc. Now I
know. It takes about 3–4 months to complete this 5-course specialization.

![](https://cdn-images-1.medium.com/max/1000/1*lnWyrQUs6d2CWbG469I8Ag.jpeg)

### What’s Next

That’s all for the first article. Please give me some claps and I will see you
in Part 1!
