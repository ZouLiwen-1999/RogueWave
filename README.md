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
We release the RWD-10K dataset which has 10191 rogue wave images. All the images are named as aXeYuZ where X, Y and Z are the orresponding parameter values in the initial equation. One .jpg image file corresponds to one .xml file which contains the bounding boxes annotation of the origin images for the rogue wave detection. You can see more details about the dataset in our paper above. You can download the RWD-10K dataset [here](https://drive.google.com/file/d/1CdpY5Xco4TnRY0DIryRbhexJB_dTsDGA/view?usp=sharing). If you use this dataset for your research, please cite our paper. 

Once you download the RWD-10K dataset, create the following folders and put the images at `RogueWaves/images` and put the xml files at `RogueWaves/xmls` and run
```
python 0_gencsv.py
```
And the data splits are saved in `RogueWave/Annotations`.

1.Train
cd RogueWave/ and run
```
python 1_train.py
```
And the trained models are saved in `RogueWaves/snapshots`.


2.Model convert
---
once you cd RogueWave/ and run
```
python 2_convert.py
```
to convert the model saved in `RogueWaves/model` for testing.

3.Test and Evaluate
---
run 
```
python 3_test2input.py
```
for putting the test images in `RogueWave/input`, then run

```
python 4_test.py
```
for testing. Finally, for evalutation you can run
```
python 5_eval.py
```

4.Inference
--
For inference, you can run
```
python 6_predict.py
```
for detecting your images.

5.Citation
--
```
@article{zou2021rw,
  title={Measuring the rogue wave pattern triggered from Gaussian perturbations by deep learning},
  author={Liwen Zou, XinHang Luo, Delu Zeng, Liming Ling and Li-Chen Zhao},
  booktitle={PRE},
  year={2021}
}
```

6.Acknowledgements
--
Part of codes are reused from the [SKU110K](https://github.com/eg4000/SKU110K_CVPR19). Thanks to Eran Goldman et al. for the codes of SKU-110K detector.

7.Contact
--
Liwen Zou([3395473905@qq.com](3395473905@qq.com))

