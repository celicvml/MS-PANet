## Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* torchvision
* numpy
* skimage
* h5py
* MATLAB

## Test
python test.py --checkpoint models/MSBDN-RDFF/1/MSBDN_epoch_50.pkl

## Train
python train.py --dataset datasets/RESIDE/ --lr 1e-4 --batchSize 1 --model MSBDN-DFF-v1-1 --name MSBDN-DFF
## dataset
dataset CDPD-55000 https://pan.baidu.com/s/1usmodCY9W9m0uGc5ZVPx6w?pwd=CVML 提取码: CVML 

## paper
paper：https://www.mdpi.com/2313-433X/12/5/189
