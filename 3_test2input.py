import csv
from PIL import Image
import os
obj_list = []
with open('../data/eu/Annotations/annotations_test.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        obj_list.append(row)

if not os.path.exists('../data/eu/input/images-optional/'):
    os.makedirs('../data/eu/input/images-optional/')
if not os.path.exists('../data/eu/input/ground-truth/'):
    os.makedirs('../data/eu/input/ground-truth/')
if not os.path.exists('../data/eu/input/detection-results/'):
    os.makedirs('../data/eu/input/detection-results/')
#存图片和注释
img_now=' '
for obj in obj_list:
	img=obj[0]
	box=[obj[1],obj[2],obj[3],obj[4]]
	lable=obj[5]
	if img!=img_now:
		image=Image.open('../data/eu/images/'+img)
		image.save('../data/eu/input/images-optional/'+img)
		img_now=img
	with open('../data/eu/input/ground-truth/'+img[:-4]+'.txt','a+') as f:
		f.write(str(lable)+' '+box[0]+' '+box[1]+' '+box[2]+' '+box[3]+' '+'\n')
	

