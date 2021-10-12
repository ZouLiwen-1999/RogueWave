import os
import keras

import sys
sys.path.insert(0, '../')

# import keras_retinanet
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from object_detector_retinanet.keras_retinanet.utils.colors import label_color
# from object_detector_retinanet.keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2

import random
import numpy as np
import time

# import copy

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
md='02'
model_path = '../data/eu/model/101_'+str(md)+'.h5'
label_map=[['background',0],['1',1],['2',2],['3',3],['4',4],['5',5]]
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')

datadir='../data/eu/input/images-optional/'
resultdir='../data/eu/input/detection-results/'
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
all_img=os.listdir(datadir)
N=len(all_img)
n=0
for img_ in all_img:
	n+=1
	print(n,'/',N)    
	# load image
	img=datadir+img_
	image = read_image_bgr(img)
	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	boxes /= scale
	# visualize detections
	with open(resultdir+img_[:-4]+'.txt','w+') as f:
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
			if score < 0.5:
				break
			f.write(label_map[label][0]+' '+str(score)+' '+str(int(box[0]))+' '+str(int(box[1]))+' '+str(int(box[2]))+' '+str(int(box[3]))+'\n')
	# break
      
