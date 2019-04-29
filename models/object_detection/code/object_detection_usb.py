"""A demo to classify Raspberry Pi camera stream."""
import argparse
import time

import numpy as np
import os
import datetime

import edgetpu.detection.engine
import cv2
from PIL import Image

def main():
    os.chdir('/home/pi/DeepPiCar/models/object_detection')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=False)
    parser.add_argument(
      '--label', help='File path of label file.', required=False)
    args = parser.parse_args()
    
    #args.model = 'test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite'
    args.model = 'data/model_result/road_signs_quantized.tflite'
    args.label = 'data/model_result/road_sign_labels.txt'
        
    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    # initialize open cv
    IM_WIDTH = 640
    IM_HEIGHT = 480
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,IM_HEIGHT-10)
    fontScale = 1
    fontColor = (255,255,255)  # white
    boxColor = (0,0,255)   # RED?
    boxLineWidth = 1
    lineType = 2
    
    annotate_text = ""
    annotate_text_time = time.time()
    time_to_show_prediction = 1.0 # ms
    min_confidence = 0.20
    
    # initial classification engine
    engine = edgetpu.detection.engine.DetectionEngine(args.model)
    elapsed_ms = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (IM_WIDTH,IM_HEIGHT))
    
    
    try:
        while camera.isOpened():
            try:
                start_ms = time.time()
                ret, frame = camera.read() # grab a frame from camera
                if ret == False :
                    print('can NOT read from camera')
                    break

                input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB color space
                img_pil = Image.fromarray(input)
                #input = cv2.resize(input, (width,height))
                start_tf_ms = time.time()
                results = engine.DetectWithImage(img_pil, threshold=min_confidence, keep_aspect_ratio=True,
                                   relative_coord=False, top_k=5)
                end_tf_ms = time.time()
                elapsed_tf_ms = end_tf_ms - start_ms
                
                if results :
                    for obj in results:
                        
                        print("%s, %.0f%% %s %.2fms" % (labels[obj.label_id], obj.score *100, obj.bounding_box, elapsed_tf_ms * 1000))
                        box = obj.bounding_box
                        coord_top_left = (int(box[0][0]), int(box[0][1]))
                        coord_bottom_right = (int(box[1][0]), int(box[1][1]))
                        cv2.rectangle(img, coord_top_left, coord_bottom_right, boxColor, boxLineWidth)
                        annotate_text = "%s, %.0f%%" % (labels[obj.label_id], obj.score * 100)
                        coord_top_left = (coord_top_left[0],coord_top_left[1]+15)
                        cv2.putText(img, annotate_text, coord_top_left, font, fontScale, boxColor, lineType )
                    print('------')
                else:
                    print('No object detected')

                # Print Frame rate info
                elapsed_ms = time.time() - start_ms
                annotate_text = "%.2f FPS, %.2fms total, %.2fms in tf " % (1.0 / elapsed_ms, elapsed_ms*1000, elapsed_tf_ms*1000)
                print('%s: %s' % (datetime.datetime.now(), annotate_text))
                cv2.putText(img, annotate_text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                
                out.write(img)
                    
                cv2.imshow('Detected Objects', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # catch it and don't exit the while loop
                print('In except')
                traceback.print_exc()

    finally:
        print('In Finally')
        camera.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
