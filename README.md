# Lightweight-Deep-CNN-for-Natural-Image-Matting-via-Similarity-Preserving-Knowledge-Distillation
## Introduction
  **Accepted at IEEE Signal Processing Letters 2020**
  
<p align="center">
  <img src="images/image.png" width="350" title="Original Image"/>
  <img src="images/spatial+channel.png" width="350" title="spatial+channel method"/>
</p>

  Official implementation of the paper "Lightweight Deep CNN for Natural Image Matting via Similarity Preserving Knowledge Distillation" [[paper](https://ieeexplore.ieee.org/document/9269400)]
 
 Donggeun Yoon, Jinsun Park, Donghyeon Cho
 
## Requirement
- python3
- pytorch
- torchvision
- OpenCV
- numpy
- scipy
- tensorboard
- tqdm

## Performace
### note
- training epochs=30
- DIM-student's parameters are 20.2% of DIM-teacher's

Here is the results of DIM-student with and without knowledge distillation on the Adobe Image Matting Dataset:
|Methods|SAD|MSE|Grad|Conn|
|---|---|---|---|---|
|without KD|121.77|0.058|75.36|129.55|
|batch similarity|124.43|0.055|74.36|132.25|
|spatial similarity|95.40|0.039|54.71|100.92|
|channel similarity|94.76|0.038|56.36|100.36|
|spatial+channel|84.37|0.034|47.63|89.35|
|batch+spatial+channel|91.30|0.037|56.20|97.20|

## Dataset
1. Please contact authors requesting for the Adobe Image Matting dataset.
2. Download images from the COCO and Pascal VOC datasets in folder `data` and Run the following command to composite images.  
```bash
$ python pre_process.py
```
3. Run the following command to seperate the composited datasets with training set and valid set.
```bash
$ python data_gen.py
```
### Training
Download pretrained [teacher model](https://github.com/foamliu/Deep-Image-Matting-PyTorch) before train and place in folder `pretrained`.
Run the following command to train with batch, spatial, channel similarity preserving knowledge distillation.
```bash
$ python train.py --batch-size 16 --KD_type batch,spatial,channel --feature_layer [1,2,3,4] --KD_weight [1,1,1]
```
### testing
Run the following command to evaluate `BEST_checkpoint.tar`.
```bash
$ python test.py
```
    
## Acknowledgement
The code is built on [Deep image matting (pytorch)](https://github.com/foamliu/Deep-Image-Matting-PyTorch). Thanks to authors for sharing the codes.

## Citation

```
@ARTICLE{9269400,
  author={D. {Yoon} and J. {Park} and D. {Cho}},
  journal={IEEE Signal Processing Letters}, 
  title={Lightweight Deep CNN for Natural Image Matting via Similarity-Preserving Knowledge Distillation}, 
  year={2020}
}
```
