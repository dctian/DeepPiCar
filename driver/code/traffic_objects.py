from threading import Timer
import logging


class TrafficObjects(object):

    def set_car_state(self, car_state):
        pass

    def is_close_by(self, obj, frame_height):
        # default: if a sign is 10% of the height of frame
        obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        return obj_height / frame_height > 0.10

class RedTrafficLight(TrafficObjects):

    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed'] = 0


class GreenTrafficLight(TrafficObjects):

    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')


class Person(TrafficObjects):

    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        car_state['speed'] = 0


class SpeedLimit(TrafficObjects):

    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_car_state(self, car_state):
        car_state['speed_limit'] = self.speed_limit


class StopSign(TrafficObjects):
    """
    Stop Sign object would wait
    """

    def __init__(self, wait_time_in_sec=2):
        self.inStopSignWaitMode = False
        self.hasStopped = False
        self.waitTimeInSec = wait_time_in_sec
        self.timer = Timer(wait_time_in_sec, self.wait_done)

    def set_car_state(self, car_state):
        if self.inStopSignWaitMode:
            logging.debug('stop sign: 2) still waiting')
            # wait for 2 second before proceeding
            car_state['speed'] = 0
            return

        if not self.hasStopped:
            logging.debug('stop sign: 1) just detected')

            car_state['speed'] = 0
            self.inStopSignWaitMode = True
            self.hasStopped = True
            self.timer.start()
            return

    def wait_done(self):
        logging.debug('stop sign: 3) finished waiting for %d seconds' % self.waitTimeInSec)
        self.inStopSignWaitMode = False

    def clear(self):
        if self.hasStopped:
            logging.debug("stop sign: 4) no more stop sign detected")
            self.hasStopped = False
            self.inStopSignWaitMode = False  # may not need to set this
