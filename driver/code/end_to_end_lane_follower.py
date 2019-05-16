import cv2
import numpy as np
import logging
import math
from keras.models import load_model
from hand_coded_lane_follower import HandCodedLaneFollower

_SHOW_IMAGE = False


class EndToEndLaneFollower(object):

    def __init__(self,
                 car=None,
                 model_path='/home/pi/DeepPiCar/models/lane_navigation/data/model_result/lane_navigation.h5'):
        logging.info('Creating a EndToEndLaneFollower...')

        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model(model_path)

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        show_image("orig", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("curr_steering_angle = %d" % self.curr_steering_angle)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

    def compute_steering_angle(self, frame):
        """ Find the steering angle directly based on video frame
            We assume that camera is calibrated to point to dead center
        """
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = self.model.predict(X)[0]

        logging.debug('new steering angle: %s' % steering_angle)
        return int(steering_angle + 0.5) # round the nearest integer


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


############################
# Test Functions
############################
def test_photo(file):
    lane_follower = EndToEndLaneFollower()
    frame = cv2.imread(file)
    combo_image = lane_follower.follow_lane(frame)
    show_image('final', combo_image, True)
    logging.info("filename=%s, model=%3d" % (file, lane_follower.curr_steering_angle))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_file):
    end_to_end_lane_follower = EndToEndLaneFollower()
    hand_coded_lane_follower = HandCodedLaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_end_to_end.avi" % video_file, video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            frame_copy = frame.copy()
            logging.info('Frame %s' % i)
            combo_image1 = hand_coded_lane_follower.follow_lane(frame)
            combo_image2 = end_to_end_lane_follower.follow_lane(frame_copy)

            diff = end_to_end_lane_follower.curr_steering_angle - hand_coded_lane_follower.curr_steering_angle;
            logging.info("desired=%3d, model=%3d, diff=%3d" %
                          (hand_coded_lane_follower.curr_steering_angle,
                          end_to_end_lane_follower.curr_steering_angle,
                          diff))
            video_overlay.write(combo_image2)
            cv2.imshow("Hand Coded", combo_image1)
            cv2.imshow("Deep Learning", combo_image2)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_video('/home/pi/DeepPiCar/models/lane_navigation/data/images/video01')
    #test_photo('/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_100_084.png')
    # test_photo(sys.argv[1])
    # test_video(sys.argv[1])
