# Integral-Human-Pose-Regression-for-3D-Human-Pose-Estimation
<p align="center">
<img src="https://cv.snu.ac.kr/research/Integral3DHumanPose/figs/1.png" width="400" height="250"> <img src="https://cv.snu.ac.kr/research/Integral3DHumanPose/figs/2.png" width="400" height="250">
</p>

## Introduction

This repo is **[PyTorch](https://pytorch.org/)** implementation of **[Integral Human Pose Regression (ECCV 2018)](https://arxiv.org/abs/1711.08229)** of MSRA for **3D human pose estimation** from a single RGB image.

**What this repo provides:**
* [PyTorch](https://pytorch.org/) implementation of [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229).
* Flexible and simple code.
* Dataset pre-processing codes for **[MPII](http://human-pose.mpi-inf.mpg.de/)** and **[Human3.6M](http://vision.imar.ro/human3.6m/description.php)** dataset.

## Dependencies
* [PyTorch](https://pytorch.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* [COCO API](https://github.com/cocodataset/cocoapi)

This code is tested under Ubuntu 16.04, CUDA 9.0, cuDNN 7.1 environment with two NVIDIA 1080Ti GPUs.

Python 3.6.5 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- data
|-- common
|-- main
|-- tool
`-- output
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `common` contains kernel codes for 3d human pose estimation system.
* `main` contains high-level codes for training or testing the network.
* `tool` contains Human3.6M dataset preprocessing code.
* `output` contains log, trained models, visualized outputs, and test result.

### Data
You need to follow directory structure of the `data` as below.
```
${POSE_ROOT}
|-- data
|-- |-- MPII
|   `-- |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg
|-- |-- Human36M
|   `-- |-- data
|       |   |-- s_01_act_02_subact_01_ca_01
|       |   |-- s_01_act_02_subact_01_ca_02
```
* In the `tool`, run `preprocess_h36m.m` to preprocess Human3.6M dataset. It converts videos to images and save meta data for each frame. `data` in `Human36M` contains the preprocessed data.
* Use MPII dataset preprocessing code in my [TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose) git repo
* You can change default directory structure of `data` by modifying `$DATASET_NAME.py` of each dataset folder.

### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- log
|-- |-- model_dump
|-- |-- result
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.
* You can change default directory structure of `output` by modifying `main/config.py`.

## Running code
### Start
* In the `main/config.py`, you can change settings of the model including dataset to use, network backbone, and input size and so on.

### Train
In the `main` folder, set training set in `config.py`. Note that `trainset` must be `list` type and `0th` dataset is the reference dataset.

In the `main` folder, run
```bash
python train.py --gpu 0-1
```
to train the network on the GPU 0,1. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0-1 --continue
```
`--gpu 0,1` can be used instead of `--gpu 0-1`.

### Test
In the `main` folder, set testing set in `config.py`. Note that `testset` must be `str` type.

Place trained model at the `output/model_dump/`.

In the `main` folder, run 
```bash
python test.py --gpu 0-1 --test_epoch 16
```
to test the network on the GPU 0,1 with 16th epoch trained model. `--gpu 0,1` can be used instead of `--gpu 0-1`.

## Results
Here I report the performance of the model from this repo and [the original paper](https://arxiv.org/abs/1711.08229). Also, I provide pre-trained 3d human pose estimation models.
 
### Results on Human3.6M dataset
The tables below are PA MPJPE and MPJPE on Human3.6M dataset. Provided `config.py` file is used to achieve below results. It's currently slightly worse than the performance of the original paper, however I'm trying to achieve the same performance. I think training schedule has to be changed.

#### Protocol 1 (training subjects: 1,5,6,7,8,9, testing subjects: 11)
Protocol 1 model and result will be available soon!!

#### Protocl 2 (training subjects: 1,5,6,7,8, testing subjects: 9, 11), PA MPJPE
The PA MPJPEs of the paper are from protocol 1, however, note that protocol 2 uses smaller training set.
| Methods | Dir. | Dis. | Eat | Gre. | Phon. | Pose | Pur. | Sit. | Sit D. | Smo. | Phot. | Wait | Walk | Walk D. | Walk P. | Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| my repo | 39.0 | 38.6 | 44.1 | 42.5 | 40.6 | 35.3 | 38.2 | 49.9 | 59.4 | 41.00 | 46.1 | 37.6 | 30.3 | 40.8 | 35.5 | 41.5 |
| [original paper](https://arxiv.org/abs/1711.08229) | 36.9 | 36.2 | 40.6 | 40.4 | 41.9 | 34.9 | 35.7 | 50.1 | 59.4 | 40.4 | 44.9 | 39.0 | 30.8 | 39.8 | 36.7 | 40.6 |

#### Protocl 2 (training subjects: 1,5,6,7,8, testing subjects: 9, 11), MPJPE
| Methods | Dir. | Dis. | Eat | Gre. | Phon. | Pose | Pur. | Sit. | Sit D. | Smo. | Phot. | Wait | Walk | Walk D. | Walk P. | Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| my repo | 50.8 | 52.3 | 54.8 | 57.9 | 52.8 | 47.0 | 52.1 | 62.0 | 73.7 | 52.6 | 58.3 | 50.4 | 40.9 | 54.1 | 45.1 | 53.9 |
| [original paper](https://arxiv.org/abs/1711.08229) | 47.5 | 47.7 | 49.5 | 50.2 | 51.4 | 43.8 | 46.4 | 58.9 | 65.7 | 49.4 | 55.8 | 47.8 | 38.9 | 49.0 | 43.8 | 49.6 |

* Pre-trained model of protocol 2 [[model](https://cv.snu.ac.kr/research/Integral3DHumanPose/model/snapshot_16.pth.tar)]

## Acknowledgement
This repo is largely modified from [Original PyTorch repo of IntegralHumanPose](https://github.com/JimmySuen/integral-human-pose).

## Reference
[1] Sun, Xiao and Xiao, Bin and Liang, Shuang and Wei, Yichen. "Integral human pose regression". ECCV 2018.
