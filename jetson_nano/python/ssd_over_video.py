# SSD using mobilenet v1 based on COCO
# Example of using this file:
# python ssd_over_video.py -t <trt_graph_path> -l <pbtxt_path> \
#                          -i <video_path> -n <total_number_of_classes> \
#                          -c <confidence>
# Examples (default is 2 so you can ommit n classes for this example)
#
# python ssd_over_video.py -t ssd_mobilenet_v1_coco_tf_trt.pb \
# -l ssd_fox_badger.pbtxt -i my_video.mp4 \
# -n 2 -c 0.4
#
# python ssd_over_video.py -t ssd_mobilenet_v1_coco_tf_trt.pb \
# -l ssd_fox_badger.pbtxt -n 2 -c 0.4

import os
import cv2
import argparse
import imutils
import time
import numpy as np
import tensorflow as tf
from imutils.video import FPS
from imutils.video import VideoStream
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.util import deprecation
from object_detection.utils import label_map_util


# Load the graph
def loadTRTGraph(graphFile):
    # open the graph file
    with tf.io.gfile.GFile(graphFile, "rb") as f:
        # instantiate the GraphDef class and read the graph
        graphDef = tf.compat.v1.GraphDef()
        graphDef.ParseFromString(f.read())

    # return the graph    
    return graphDef

# turn off the deprecation warnings and logs to 
# keep the console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trt_graph", required=True,
    help="Path of the trt graph file")
ap.add_argument("-l", "--labels", required=True,
    help="path to the pbtxt file")
ap.add_argument("-i", "--input",
    help="path to input video file")
ap.add_argument("-n", "--num_classes", type=int, required=True,
    help="number of classes to detect", default=2)
ap.add_argument("-c", "--conf", type=float, default=0.5,
    help="below this number detections will be ignored")
args = vars(ap.parse_args())


trtfile    = args["trt_graph"]
labelsfile = args["labels"] 
confidence = float(args["conf"])
inputfile  = args["input"]
numclasses = int(args["num_classes"])

print("[INFO] - check your inputs below...")
print(f"[INFO] - tensor rt graph file = {trtfile}")
print(f"[INFO] - labels file          = {labelsfile}")
print(f"[INFO] - confidence threshold = {confidence}")
print(f"[INFO] - input video          = {inputfile}")
print(f"[INFO] - total of classes     = {numclasses}")
# be consistent on random number generation
np.random.seed(42) 

color_list = np.random.uniform(0, 255, size=(2, 3))
labelMap   = label_map_util.load_labelmap(labelsfile)
categories = label_map_util.convert_label_map_to_categories(labelMap, 
                                                            max_num_classes=numclasses,
                                                            use_display_name=True)

# load the TRT graph
print("[INFO] - loading TRT graph...")
trtGraph = loadTRTGraph(args["trt_graph"])

# instantiate the ConfigProto class, enable GPU usage growth, create
# TensorFlow session, and import the TRT graph into the session
print("[INFO] - initializing TensorFlow session...")
tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tf_sess = tf.compat.v1.Session(config=tfConfig)
tf.import_graph_def(trtGraph, name="")

# list of inputs and outputs 
# This list was taken when you got write the trt file
input_names  = ['image_tensor:0']
output_names = ['detection_boxes',
                'detection_classes', 
                'detection_scores',
                'num_detections']


tf_input          = tf_sess.graph.get_tensor_by_name(input_names[0])
# output tensors
tf_boxes          = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
tf_classes        = tf_sess.graph.get_tensor_by_name(output_names[1] + ':0')
tf_scores         = tf_sess.graph.get_tensor_by_name(output_names[2] + ':0')
tf_num_detections = tf_sess.graph.get_tensor_by_name(output_names[3] + ':0')


# initialize variables to store frame dimensions
H = None
W = None

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    #vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(inputfile)


# start the frames per second throughput estimator
fps = FPS().start()

while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if inputfile is not None and frame is None:
        break

    # resize the frame to have a maximum width of 300 pixels
    #frame = imutils.resize(frame, width=300)

    # check to see if the frame dimensions are not set
    if W is None or H is None:
        # set the frame dimensions
        (H, W) = frame.shape[:2]

    # copy to preserve
    out = frame.copy()
    # get the width and height of the image
    H, W = out.shape[:2]
    # change format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # flat array
    image = np.expand_dims(image, axis=0)

    # Run the inference
    output_list = [tf_scores, tf_boxes, tf_classes, tf_num_detections]
    input_dict  = {tf_input: image}
    predictions = tf_sess.run(output_list, feed_dict=input_dict)
    scores, boxes, classes = predictions[:3]

    # Remove the dimension (always is 1 image)
    boxes = np.squeeze(boxes)                   # same as boxes[0], removes a dim
    scores = np.squeeze(scores)                 # same as scores[0]
    classes = np.squeeze(classes)               # same as classes[0]

    # Create the category list array
    categoryIdx = label_map_util.create_category_index(categories)

    # process all images
    for box, score, label in zip(boxes, scores, classes):
        if score < confidence:
            continue
        # scale box to image coordinates
        startY, startX, endY, endX = box

        # fix sizes and do some calculations
        startX      = int(startX * W)
        startY      = int(startY * H)
        endX        = int(endX * W)
        endY        = int(endY * H)
        w           = endX - startX
        h           = endY - startY
        start_point = (startX, startY) 
        end_point   = (endX, endY) 
        color_num   = categoryIdx[label]['id'] - 1
        color       = color_list[color_num] 

        # Display overlay
        # Enable line below and comment next ones if you only want a rectangle
        out = cv2.rectangle(out, start_point, end_point, color, 4)
        # Comment line above and uncomment below if you want only alpha blending
        rect = np.full((h,w,3),color, dtype='uint8')
        alpha       = 0.8
        beta        = 1 - alpha
        crop        = out[startY:endY,startX:endX]
        cv2.addWeighted(crop, alpha, rect, beta, 0, crop)

        # display class index and score
        lbl = categoryIdx[label]['name']
        data = lbl + ': ' + str(np.round(score *100,2)) +'%'
        cv2.putText(out, data, (startX + 5, startY + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(out, data, (startX + 5, startY + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 2)
    # show the output frame
    cv2.imshow("Jetson NanoCam", out)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()