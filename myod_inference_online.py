# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:38:56 2017

@author: meldo
"""
from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append('..')
sys.path.append('/dl/research/tfod')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to inference model base.
MODEL_BASE = '/dl/research/myod/inference'
PATH_TO_IMAGES = MODEL_BASE + '/images'
PATH_TO_MODELS = MODEL_BASE + '/models'

# Path to the images.
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_IMAGES, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

##################
# Download Model #
##################
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = MODEL_BASE + '/models/data/mscoco_label_map.pbtxt'

# Num of classes
NUM_CLASSES = 90

# What model to download.
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph for the object detection.
PATH_TO_CKPT = os.path.join(PATH_TO_MODELS, MODEL_NAME + '/frozen_inference_graph.pb')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, PATH_TO_MODELS)

################################################
# Load a (frozen) Tensorflow model into memory #
################################################
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#####################
# Loading label map #
#####################
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###############
# Helper code #
###############
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#############
# Detection #
#############
tf.logging.info('Inference begins...')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()