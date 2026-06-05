# MS-PANet

[**论文链接 / MS-PANet: Multi-Scale Spatial Pyramid Attention for Effective Drainage Pipeline Image Dehazing**](https://www.mdpi.com/2313-433X/12/5/189)


## Environment Setup
* Python 3.6
* PyTorch >= 1.1.0
* torchvision
* numpy
* skimage
* h5py
* MATLAB

## Dataset Download
dataset CDPD-55000 https://pan.baidu.com/s/1usmodCY9W9m0uGc5ZVPx6w?pwd=CVML 提取码: CVML 

The dataset is available at the links above.


## Train your own model
you can retrain the model by yourself with following command.

```bash
python train.py --dataset datasets/RESIDE/ --lr 1e-4 --batchSize 1 --model MSBDN-DFF-v1-1 --name MSBDN-DFF
python test.py --checkpoint models/MSBDN-RDFF/1/MSBDN_epoch_50.pkl
```

If you find our repo useful, please give us a star and cite:

```bash

@article{li_MS-PANet,
    author = {Cheng Li and Xiaoyan Duan and Zhongbo Jiang and Yong Ding and Qiang Li and Zhen Tang and Feng Yang},
    title = {MS-PANet: Multi-Scale Spatial Pyramid Attention for Effective Drainage Pipeline Image Dehazing},
    journal = {Journal of Imaging},
    year = {2026},
    volume = {12},
    number = {5},
    pages = {189},
    doi = {10.3390/jimaging12050189}
}
