# [CVPR 2023] Curricular Contrastive Regularization for Physics-aware Single Image Dehazing

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/curricular-contrastive-regularization-for/image-dehazing-on-sots-indoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-indoor?p=curricular-contrastive-regularization-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/curricular-contrastive-regularization-for/image-dehazing-on-sots-outdoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-outdoor?p=curricular-contrastive-regularization-for)

This is the official PyTorch codes for the paper:  
>**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing**<br>  [Yu Zheng](https://github.com/YuZheng9), [Jiahui Zhan](https://github.com/zhanjiahui), [Shengfeng He](http://www.shengfenghe.com/), [Junyu Dong](https://it.ouc.edu.cn/djy_23898/main.htm), [Yong Du<sup>*</sup>](https://www.csyongdu.cn/) （ * indicates corresponding author)<br>
>IEEE/CVF Conference on Computer Vision and Pattern Recognition

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
conda install pytorch=1.10.1 torchvision torchaudio cudatoolkit=11.4 -c pytorch
pip install -r requirements.txt
```

##  Training and Evaluation

### Prepare dataset for evaluation


You can download the pretrained models and datasets on [BaiduNetdisk](https://pan.baidu.com/s/1CPpdWz60HsTM2w16mq2S-w?pwd=b76u).

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
Since our training dataset contains numerous negative samples, which has led to an oversized dataset, we have only uploaded the `ITS` training dataset. However, users can create additional datasets using the `create_lmdb` function, where they can define the number of negative samples and the types of existing dehazers used.

You can refer to our `create_lmdb` and use six images recovered from different networks as six negative samples of different difficulties.

The final file path will be arranged as (please check it carefully)：

```
data
   |-ITS
      |- ITS.lmdb
   |- ...(dataset name)
```

### Train

train network on `ITS` 

```
python main.py --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='its_train' --testset='its_test' --steps=1000000 --eval_step=5000 --clcrloss --clip
```
train network on `OTS` 

```
python main.py --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='ots_train' --testset='ots_test' --steps=1500000 --eval_step=5000 --clcrloss --clip
```

### Evaluation
<details>
<summary>SOTS-indoor (click to expand) </summary>

`python dehaze.py -d indoor`
</details>

<details>
<summary>SOTS-outdoor (click to expand) </summary>

`python dehaze.py -d outdoor`
</details>


See `python dehaze.py -h ` for list of optional arguments


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
