import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import xml.etree.ElementTree as ET
import random
from random import randint
import csv
import copy
from random import sample
path=os.path.join
listdir=os.listdir

label_map=[['background',0],['1',1],['2',2],['3',3],['4',4],['5',5]]
data_path='./'
img_path=data_path+'images/'
xml_path=data_path+'xmls/'
csv_path=data_path+'Annotations/'
snapshots_path=data_path+'snapshots/'
logs_path=data_path+'logs/'
models_path=data_path+'model/'

if not os.path.exists(csv_path):
	os.makedirs(csv_path)
if not os.path.exists(snapshots_path):
	os.makedirs(snapshots_path)
if not os.path.exists(logs_path):
	os.makedirs(logs_path)
if not os.path.exists(models_path):
	os.makedirs(models_path)
	
def dump_csv(objects_list):
	annotations=[]
	obj=0
	for i in range(len(objects_list)):
		img=objects_list[i]['name']
		width=objects_list[i]['width']
		height=objects_list[i]['height']
		bndbox=objects_list[i]['boxes']
		labels=objects_list[i]['labels']
		if len(bndbox)>0:
			for j in range(len(bndbox)):
				annotations.append([])
				annotations[obj].append(img)
				annotations[obj].append(bndbox[j][0])
				annotations[obj].append(bndbox[j][1])
				annotations[obj].append(bndbox[j][2])
				annotations[obj].append(bndbox[j][3])
				annotations[obj].append(labels[j])
				annotations[obj].append(width)
				annotations[obj].append(height)
				obj=obj+1
		else:
			annotations.append([])
			annotations[obj].append(img)
			annotations[obj].append(None)
			annotations[obj].append(None)
			annotations[obj].append(None)
			annotations[obj].append(None)
			annotations[obj].append(None)
			annotations[obj].append(width)
			annotations[obj].append(height)
			obj=obj+1
	return annotations

#将xml转化为字典
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path) 
    root = tree.getroot()
    boxes = list()
    labels = list()
    for obj in root.iter('object'):
        label = label_map[int(obj.find('name').text)][0]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) 
        ymin = int(bbox.find('ymin').text) 
        xmax = int(bbox.find('xmax').text) 
        ymax = int(bbox.find('ymax').text) 
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return {'boxes': boxes, 'labels': labels}

#获得某张图片的三个参数值
def get_aeu(name):#加后缀的文件名
	a_no=0
	e_no=0
	u_no=0
	for char in name:
		if char!='e':
			e_no+=1
		else:
			break
	for char in name:
		if char!='u':
			u_no+=1
		else:
			break
	a=float(name[a_no+1:e_no])
	e=float(name[e_no+1:u_no])
	u=float(name[u_no+1:-4])
	return {'a':a,'e':e,'u':u,'a_no':a_no,'e_no':e_no,'u_no':u_no}


#划分训练验证测试集
imgs=listdir(img_path)
random.shuffle(imgs)
data={}
data['train']=imgs[:int(0.6*len(imgs))]
data['val']=imgs[int(0.6*len(imgs)):int(0.8*len(imgs))]
data['test']=imgs[int(0.8*len(imgs)):]
data['all']=imgs

#打好图片类别（按u分为5类）
cls_root='u'
num_cls=5
for split in ['test','all','train','val',]:
	image_list=[]
	with open(data_path+split+'.txt','w') as f:
	    n=0
	    N=len(data[split])
	    for line in data[split]:
	        print(split,n,'/',N)
	        n+=1
	        if line[0]!='.':
	        	key=get_aeu(line)[cls_root]
	        	for i in range(num_cls):
	        		if key<=(i+1)*10 and key>i*10:
	        			label=str(i+1)
	        			break
	        	f.write(line+','+label+'\n')
	        	image = Image.open(img_path+line)
	        	info=parse_annotation(xml_path+line.replace('.jpg','.xml'))
	        	info['name']=line
	        	info['width']=image.width
	        	info['height']=image.height
	        	image_list.append(info)
	anno=dump_csv(image_list)
	with open(csv_path+'annotations_'+split+'.csv', 'w') as csvfile:
	    writer=csv.writer(csvfile)
	    for row in anno:
	        writer.writerow(row)

with open(csv_path+'class_mappings.csv', 'w') as csvfile:
	    writer=csv.writer(csvfile)
	    for row in label_map:
	        writer.writerow(row)