import os
import keras
import sys
sys.path.insert(0, '../')
# import keras_retinanet
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from object_detector_retinanet.keras_retinanet.utils.colors import score_color
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
import copy
#from NMS import nms

def kBoxes(boxes,k=0.5):
	for box in boxes:
		mid_x=(box[0]+box[2])/2
		mid_y=(box[1]+box[3])/2
		box[0]=mid_x+k*(box[0]-mid_x)
		box[2]=mid_x+k*(box[2]-mid_x)
		box[1]=mid_y+k*(box[1]-mid_y)
		box[3]=mid_y+k*(box[3]-mid_y)
	return boxes

def kBox(box,k=0.5):
	mid_x=(box[0]+box[2])/2
	mid_y=(box[1]+box[3])/2
	box[0]=mid_x+k*(box[0]-mid_x)
	box[2]=mid_x+k*(box[2]-mid_x)
	box[1]=mid_y+k*(box[1]-mid_y)
	box[3]=mid_y+k*(box[3]-mid_y)
	return box

md=input('Input model id:')
model_path = '../data/eu/model/101_'+str(md)+'.h5'
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')
label_map=[['background',0],['1',1],['2',2],['3',3],['4',4],['5',5]]

sets='input/images-optional'
sets2='images-optional'
all_images=os.listdir('../data/eu/'+sets)
A_num=0#记录九十多分的框
B_num=0
C_num=0
D_num=0
E_num=0#记录五十多分的框
winner=[]
	# load image
n=0
N=len(all_images)
for img in all_images:
	image = read_image_bgr('../data/eu/'+sets+'/'+img)
	# copy to draw 
	n+=1   
	print(n,'/',N)
	draw = image.copy()
	# draw = copy.deepcopy(image)
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)

	# process image
	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	# boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
	# print("processing time: ", time.time() - start)
	# correct for image scale
	boxes /= scale
	num=0
	# boxes_=kBoxes(boxes[0],k=0.5)
	# boxes_=boxes[0]
	# scores_=scores[0]
	# boxes_,scores_=nms(boxes_, scores_, threshold=0.3)

    # visualize detections
	for box,score,label in zip(boxes[0],scores[0],labels[0]):
		# box=kBox(box,k=0.5)
		num=num+1
		if score < 0.5:
			break
		elif label==1:
			A_num+=1
			color = (0,0,0)
		elif label==2:
			B_num+=1
			color = (255,0,0)
		elif label==3:
			C_num+=1
			color = (0,255,0)
		elif label==4:
			D_num+=1
			color = (0,0,255)
		elif label==5:
			E_num+=1
			color = (255,255,255)

		winner.append(label) 
# 		color = score_color(score)
		b = box.astype(int)
		draw_box(draw, b, color=color)
		# caption = "{} {:.3f}".format(labels_to_names[label], score)
	    # draw_caption(draw, b, caption)
	    # scores are sorted so we can break
	
	# plt.figure(figsize=(30, 30))
	# plt.axis('off')
	# plt.imshow(draw)
	# plt.show()
	if not os.path.exists('../data/eu/dets/model_'+str(md)):
		os.makedirs('../data/eu/dets/model_'+str(md))
	
	# num=len(os.listdir('./det/'+img))
	# image.save('./det/'+img+'/'+str(num)+'.jpg')
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	cv2.imwrite('../data/eu/dets/model_'+str(md)+'/'+img,draw)
	# print('num:',num)
plt.figure()
plt.hist(winner)
plt.savefig('../data/eu/dets/model_'+str(md)+'/'+'hist.jpg')
      
