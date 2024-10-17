## Getting started

### 1. Installation

YOTO is developed based on `torch==1.11.0` `mmyolo==0.6.0` and `mmdetection==3.0.0`. Check more details about `requirements` and `mmcv` in [docs/installation](./docs/installation.md).
about `requirements` and `mmcv` in [docs/installation](./docs/installation.md).

#### Clone Project 

```bash
git clone --recursive https://github.com/lisiqi-zju/YOTO
```
#### Install

```bash
pip install torch wheel -q
pip install -e .
```

### 2. Preparing Data

Please follow the instructions in [the official website](https://github.com/coco-tasks/dataset) to download the COCO-Tasks dataset.

You can organize the 'data' folder as follows:
```
data/
  ├── id2name.json
  ├── images/
  │    ├── train2014/
  │    └── val2014/
  └── coco-tasks/
       └── annotations/
            ├── task_1_train.json
            ├── task_1_test.json
            ...
            ├── task_14_train.json
            └── task_14_test.json
```
Then set the arguments `coco_path`, `refexp_ann_path` and `catid2name_path` in file `configs/tdod.json` to be the path of `data/images/`, `data/coco-tasks/annotations/` and `data/id2name.json`, respectively.


### 3. Pre-trained Models
[HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth)
Place the file in the directory './pretrained_models'.

## Training & Evaluation

We adopt the default [training](./tools/train.py) or [evaluation](./tools/test.py) scripts of [mmyolo](https://github.com/open-mmlab/mmyolo).

Training YOTO is easy:
```bash
chmod +x tools/dist_train.sh
# sample command for pre-training, use AMP for mixed-precision training
./tools/dist_train.sh configs/YOTO.py 8 --amp
```
**NOTE:** YOLO-World is pre-trained on 4 nodes with 8 GPUs per node (32 GPUs in total). For pre-training, the `node_rank` and `nnodes` for multi-node training should be specified. 

Evaluating YOTO is also easy:

```bash
chmod +x tools/dist_test.sh
./tools/dist_test.sh configs/YOTO.py path/to/weights 8
```
