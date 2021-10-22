Measuring the rogue wave pattern triggered from Gaussian perturbations by deep learning
==

This is the official implement of the RWD-Net from the paper [Measuring the rogue wave pattern triggered from Gaussian perturbations by deep learning.](https://arxiv.org/abs/2109.08909)

Requirements
==
Python 3.7
Tensorflow-gpu==1.13.1
Keras==2.1.5

Usage
==

0.RWD-10K Dataset Preparation
--
We release the RWD-10K dataset which has 10191 rogue wave images. All the images are named as $aXeYuZ$ where X, Y and Z are the value of $a$, $\epsilon$ and $\mu$ of the initial equation. One .jpg image file corresponds to one .xml file which contains the bounding boxes annotation of the origin images for the rogue wave detection. You can see more details about the dataset in our paper above. You can download the RWD-10K dataset [here.](https://drive.google.com/file/d/1CdpY5Xco4TnRY0DIryRbhexJB_dTsDGA/view?usp=sharing). If you use this dataset for your research, please cite our paper. 

Once you download the RWD-10K dataset, create the following folders and put the images at 
```
RogueWaves/images
```
and put the xml files at
```
RogueWaves/Annotations
```

1.Train
---
Create the these folders 
```
RogueWaves/logs
```
```
RogueWaves/snapshots
```
```
RogueWaves/models
```
cd RogueWave/ and run
```
1_train.py
```
The trained models are saved in
```
RogueWaves/models

```
2.Model convert
---
once you cd RogueWave/ and run
```
2_convert.py
```
to convert the model for testing.

3.




