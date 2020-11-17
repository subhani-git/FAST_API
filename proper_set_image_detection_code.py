#importing the neccesary packages
import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print('Using Tensorflow Version ' + str(tf.__version__))
import zipfile
import re
import glob
import cv2
import itertools
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
import imutils

# sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

##Argumentparser is used to run through the terminal
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', dest='input_image_path',
#                     help='image to be processed')
# parser.add_argument('-o', dest='saved_image_path',
#                     help='saved image path')
# args = parser.parse_args()

##Input and output video path
#INPUT_IMG_PATH =  args.input_image_path
#Saved_IMG_PATH = r'D:\PycharmProjects\fastapi\ssd'

# print ('input_image_path     =', INPUT_IMG_PATH)
# print ('saved_image_path     =', Saved_IMG_PATH)

##Threshold at which the detection bounding boxes will display
TH = 0.70

# Gloabl Variables
image_tensor = None
detection_boxes = None
detection_scores = None
detection_classes = None
num_detections = None

sess = None
s_category_index = None


def load_frozen_graph_into_memory(path_to_ckpt):
  ## Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph


def tensors_initialization(detection_graph, path_to_labels, num_classes):
  global s_category_index
  s_label_map = label_map_util.load_labelmap(path_to_labels)
  s_categories = label_map_util.convert_label_map_to_categories(s_label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
  s_category_index = label_map_util.create_category_index(s_categories)

  with detection_graph.as_default():
    global sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections
    sess = tf.Session(graph=detection_graph)
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  return None

def process_init(path_to_ckpt='models/shelf_resnet_frozen_inference_graph.pb',
                 path_to_labels='models/shelf_label_map.pbtxt',
                 num_classes=1):
  print('Frozen Graph Path: ', path_to_ckpt)
  print('Labels Path: ', path_to_labels)
  print('Num of Classes: ', num_classes)
  print('-----------------------------------------------------------------------')
  detection_graph = load_frozen_graph_into_memory(path_to_ckpt)
  _ = tensors_initialization(detection_graph, path_to_labels, num_classes)

  return None

def sample_detection(img):
    ##Loading image
    image = cv2.imread(img)
    print(image.shape)
    scale_percent = 100 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_np = cv2.cvtColor(cv2.resize(image,dim),cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    fetch_time = time.time()

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    sboxes = np.squeeze(boxes);
    sclasses = np.squeeze(classes).astype(np.int32);
    sscores = np.squeeze(scores);
    img_array, info = vis_util.visualize_boxes_and_labels_on_image_array(image_np, img, sboxes,sclasses,sscores,
                                                              s_category_index,min_score_thresh=TH,max_boxes_to_draw=10,
                                                              use_normalized_coordinates=True,skip_scores=False,line_thickness=2)
    image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    print(image_np.shape)

    a=cv2.imwrite(f'{os.getcwd()}/{img}', image_np)
    print(a)
    print( f'{os.getcwd()}/{img}')
    path_ = f'{os.getcwd()}/{img}'
    print(path_)
    result = []
    print(result)
    for each_tuple in info:
      print(each_tuple)
      temp = dict()

      temp['Image Name'] = each_tuple[0]
      temp['Children'] = [{'Class':each_tuple[1], 'Confidence': each_tuple[2], 'xmin': each_tuple[3], 'ymin': each_tuple[4], 'xmax': each_tuple[5], 'ymax' :each_tuple[6]}]
      print(result.append(temp))
    result.append(path_)
    return result

# paths = glob.glob(r'D:\PycharmProjects\fastapi\receive\*')
# for i in paths:
#   image_detection(i)
# # #image_detection(i)
# if os.path.isfile('./ssd/output_inference_graph.pb'):
# 	print('subhan')
# else:
#  	"no file"
# process_init('./ssd/output_inference_graph.pb/frozen_inference_graph.pb',
#              './inputs/label_map.pbtxt', 1)
# s = sample_detection('receive/cam3.jpg')
# print(s)


