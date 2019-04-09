# DeepPiCar
A deep learning autonomous car built with Raspberry Pi (3 Model B+), SunFounder PiCar-V Kit, TensorFlow, and Google's EdgeTPU Co-Processor.  

# Features of DeepPiCar
- Navigate autonomously on a winding single lane road with lane marker.  There are two implementations to this feature 
  - Use imaging recognition package to detect lane lines, and write code to change steering angle
  - Use
- Be able to recognize road signs and traffic lights, and respond according to the signs.
  - Stop sign: stop near the stop sign, wait for 2 seconds and then keep on going
  - Traffic light: stop on red light and waits for it to turn green
  - Speed sign: follow the speed sign on road until speed limit changes 

# How the Name, Deep Pi Car came about?
- First of all, it is a car based on Raspberry Pi board (Model 3+ B).  This is the heart of the whole car.  (Pi)
- Secondly, I am using a model car kit from SunFounder, which is called PiCar.  Basically, it gives me a convenient hardward and associated python API to drive the car.  (Car)
- Thirdly, I am using Google's newest Edge TPU co-processor.  It is required as the Raspberry Pi CPU alone is not powerful to run the object detection deep learning model, MobileNet SSD, in real time, which is required for traffic sign detection.  (Deep)
- Lastly, this is a nod to where the developer is located, Chicago, and its famous Deep Dish Pizza (aka Deep Pie).

# Materials and Tools Required

