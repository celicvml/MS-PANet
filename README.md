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


@Article{jimaging12050189,
AUTHOR = {Li, Ce and Duan, Xinyi and Jiang, Zhongbo and Ding, Yijing and Li, Quanzhi and Tang, Zhengyan and Yang, Feng},
TITLE = {MS-PANet: Multi-Scale Spatial Pyramid Attention for Effective Drainage Pipeline Image Dehazing},
JOURNAL = {Journal of Imaging},
VOLUME = {12},
YEAR = {2026},
NUMBER = {5},
ARTICLE-NUMBER = {189},
URL = {https://www.mdpi.com/2313-433X/12/5/189},
DOI = {10.3390/jimaging12050189}
}

