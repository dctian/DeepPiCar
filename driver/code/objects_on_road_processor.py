import cv2
import logging
import datetime
import time
import edgetpu.detection.engine
from PIL import Image

_SHOW_IMAGE = False


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=40,
                 model='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_signs_quantized_edgetpu.tflite',
                 label='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_sign_labels.txt',
                 width=640,
                 height=480):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info('Creating a ObjectsOnRoadProcessor...')

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = 0
        self.resume_driving()

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

        # initial edge TPU engine
        logging.info('Initialize Edge TPU with model %s...' % model)
        self.engine = edgetpu.detection.engine.DetectionEngine(model)
        self.min_confidence = 0.30
        self.num_of_objects = 3
        logging.info('Initialize Edge TPU with model %s done.' % model)

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        objects, final_frame = self.detect_objects(frame)
        #self.control_car(objects)

        return final_frame

    def control_car(self, objects, min_obj_height=50):
        logging.debug('processing objects...')
        if len(objects) == 0:
            logging.info('No objects detected, drive at speed limit of %s.' % self.speed_limit)
            self.resume_driving()
            return

        secs_to_wait_at_stop_sign = 2

        largest_object = find_largest_object(objects)
        if largest_object.height > min_obj_height:
            # ensure the object is close enough
            obj_label = self.labels[largest_object.label_id]
            if obj_label == 'Person':
                logging.info('Waiting for person to cross')
                self.set_speed(0)
            if obj_label == 'Red Traffic Light':
                logging.info('Stopped at red light')
                self.set_speed(0)
            elif obj_label == 'Stop Sign':
                logging.info('Stopping at stop sign for %s sec' % secs_to_wait_at_stop_sign)
                self.set_speed(0)
                time.sleep(secs_to_wait_at_stop_sign)
                self.resume_driving()
            elif obj_label == 'Speed Limit 40':
                self.speed_limit = 40
                self.resume_driving()
            elif obj_label == 'Speed Limit 25':
                self.speed_limit = 25
                self.resume_driving()
            elif obj_label == 'Green Traffic Light':
                self.resume_driving()
            return

    def resume_driving(self):
        if self.speed != self.speed_limit:
            logging.info('Current Speed = %d, New Speed = %d' % (self.speed, self.speed_limit))
            self.set_speed(self.speed_limit)

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            self.car.speed = speed

    ############################
    # Frame processing steps
    ############################
    def detect_objects(self, frame):
        logging.debug('Detecting object in the frame...')

        # call tpu for inference
        start_ms = time.time()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_RGB)
        objects = self.engine.DetectWithImage(img_pil, threshold=self.min_confidence, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=self.num_of_objects)
        if objects:
            for obj in objects:
                logging.info("%s, %.0f%% %s" % (self.labels[obj.label_id], obj.score * 100, obj.bounding_box))
                box = obj.bounding_box
                coord_top_left = (int(box[0][0]), int(box[0][1]))
                coord_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(frame, coord_top_left, coord_bottom_right, self.boxColor, self.boxLineWidth)
                annotate_text = "%s %.0f%%" % (self.labels[obj.label_id], obj.score * 100)
                coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                cv2.putText(frame, annotate_text, coord_top_left, self.font, self.fontScale, self.boxColor, self.lineType)
            logging.info('------')
        else:
            logging.info('No object detected')

        elapsed_ms = time.time() - start_ms

        annotate_summary = "%.1f FPS" % (1.0/elapsed_ms)
        logging.info(annotate_summary)
        cv2.putText(frame, annotate_summary, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)
        #cv2.imshow('Detected Objects', frame)

        return objects, frame


############################
# Utility Functions
############################
def find_largest_object(objects):
    largest_object = None
    max_height = 0
    for obj in objects:
        box = obj.bounding_box
        height = int(box[0][1] - box[0][0])
        if height >= max_height:
            largest_object = obj
    largest_object.height = max_height
    return largest_object


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


############################
# Test Functions
############################
def test_photo(file):
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread(file)
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Detected Objects', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_file):
    object_processor = ObjectsOnRoadProcessor()
    cap = cv2.VideoCapture(video_file + '.avi')

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    video_overlay = cv2.VideoWriter("%s_overlay_%s.avi" % (video_file, date_str), video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            cv2.imwrite("%s_%03d.png" % (video_file, i), frame)

            combo_image = object_processor.process_objects_on_road(frame)
            cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
            video_overlay.write(combo_image)

            cv2.imshow("Detected Objects", combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')

    # test_video('/home/pi/DeepPiCar/driver/data/car_video_orig_190411_111646/car_video_orig_190411_111646')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/red_light.jpg')
    # test_photo(sys.argv[1])
    #test_video(sys.argv[1])
