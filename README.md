# [CVPR 2023] Curricular Contrastive Regularization for Physics-aware Single Image Dehazing

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/curricular-contrastive-regularization-for/image-dehazing-on-sots-indoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-indoor?p=curricular-contrastive-regularization-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/curricular-contrastive-regularization-for/image-dehazing-on-sots-outdoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-outdoor?p=curricular-contrastive-regularization-for)

This is the official PyTorch codes for the paper:  
>**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing**<br>  [Yu Zheng](https://github.com/YuZheng9), [Jiahui Zhan](https://github.com/zhanjiahui), [Shengfeng He](http://www.shengfenghe.com/), [Junyu Dong](https://it.ouc.edu.cn/djy_23898/main.htm), [Yong Du<sup>*</sup>](https://www.csyongdu.cn/) （ * indicates corresponding author)<br>
>IEEE/CVF Conference on Computer Vision and Pattern Recognition

### Network Architecture
![Architecture](figs/framework.png)

### News

- **Apr 20, 2023:** We release training code.


## Getting started

### Install

We test the code on PyTorch 1.10.1 + CUDA 11.4 .

1. Create a new conda environment
```
conda create -n c2pnet python=3.7.11
conda activate c2pnet
```

2. Install dependencies
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

##  Training and Evaluation

### Prepare dataset for evaluation


You can download the pretrained models and datasets on [Google Drive(pretrained models and testset)](https://drive.google.com/drive/folders/1XwKEUCK__JlvoSUrD3ccCbxp7iE27-70?usp=sharing).

The final file path will be arranged as (please check it carefully)：

```
|-trained_models
     |- ITS.pkl
     |- OTS.pkl
     └─ ... (model name)
|-data
     |-SOTS
        |- indoor
           |- hazy
              |- 1400_1.png 
              |- 1401_1.png 
              └─ ... (image name)
           |- clear
              |- 1400.png 
              |- 1401.png
              └─ ... (image name)
        |- outdoor
           |- hazy
              |- 0001_0.8_0.2.jpg 
              |- 0002_0.8_0.08.jpg
              └─ ... (image name)
           |- clear
              |- 0001.png 
              |- 0002.png
              └─ ... (image name)
```

### Prepare dataset for train
Since our training dataset contains numerous negative samples, which has led to an oversized dataset, we have only uploaded the `ITS` training dataset on [BaiduNetdisk](https://pan.baidu.com/s/1vwMlP6RwIk361NQ70ZpGwg?pwd=7zkr). However, users can create additional datasets using the `create_lmdb.py`, where they can define the number of negative samples and the types of existing dehazers used.

The final file path will be arranged as (please check it carefully)：

```
data
   |-ITS
      |- ITS.lmdb
   |- ...(dataset name)
```

### Evaluation
Test C2PNet on `SOTS-indoor` dataset 

```python dehaze.py -d indoor```

Test C2PNet on `SOTS-outdoor` dataset 

```python dehaze.py -d outdoor```


See `python dehaze.py -h ` for the list of optional arguments

### Train

Train network on `ITS` dataset

```
python main.py --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='its_train' --testset='its_test' --steps=1000000 --eval_step=5000 --clcrloss --clip
```
Train network on `OTS`  dataset

```
python main.py --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='ots_train' --testset='ots_test' --steps=1500000 --eval_step=5000 --clcrloss --clip
```


## Citation
If you find our work useful for your research, please cite us:
```
@inproceedings{zheng2023curricular,
  title={Curricular Contrastive Regularization for Physics-aware Single Image Dehazing},
  author={Zheng, Yu and Zhan, Jiahui and He, Shengfeng and Dong, Junyu and Du, Yong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
