# Curricular Contrastive Regularization for Physics-aware Single Image Dehazing

## Prepare pretrained models

Before performing the following steps, please download our pretrained model first.

Our pretrained models in `trained_models` directory.

Please download file and place the models to `trained_models` directory, separately.

The directory structure will be arranged as:

```
trained_models
   |- ITS.pkl
   |- OTS.pkl
   |- NH19.pkl
   |- NH2.pkl
```

## Prepare dataset for evaluation
(SOTS)[https://sites.google.com/view/reside-dehaze-datasets/reside-v0]   
(HN-Haze2)[https://competitions.codalab.org/competitions/28032#learn_the_details]  
(Dense-Haze)[https://data.vision.ee.ethz.ch/cvl/ntire19/]

The `images` directory structure will be arranged as: (Note: please check it carefully), 

You can also refer to the `images` directory in the Anonymous Github.

```
images
   |-NH19
      |- hazy 
         |- 51.png 
         |- 52.png
         |- 53.png 
         |- 54.png
         |- 55.png 
      |- clear
         |- 51.png 
         |- 52.png
         |- 53.png 
         |- 54.png
         |- 55.png 
   |-NH21
      |- hazy 
         |- 21.png 
         |- 22.png
         |- 23.png 
         |- 24.png
         |- 25.png 
      |- clear
         |- 21.png 
         |- 22.png
         |- 23.png 
         |- 24.png
         |- 25.png 
   |-SOTS
      |- indoor
         |- hazy
            |- 1400_1.png 
            |- 1401_1.png 
            ...
         |- clear
            |- 1400.png 
            |- 1401.png
            ...
      |- outdoor
         |- hazy
            |- 0001_0.8_0.2.jpg 
            |- 0002_0.8_0.08.jpg
            ...
         |- clear
            |- 0001.png 
            |- 0002.png
            ...
```


## evaluation
<details>
<summary>SOTS-indoor (click to expand) </summary>

`python dehaze.py -d indoor`
</details>

<details>
<summary>SOTS-outdoor (click to expand) </summary>

`python dehaze.py -d outdoor`
</details>

<details>
<summary>NH-Haze2 (click to expand) </summary>

`python dehaze.py -d NH2`
</details>

<details>
<summary>Dense-Haze (click to expand) </summary>

`python dehaze.py -d dense`
</details>

See `python dehaze.py -h ` for list of optional arguments


## Train

coming soon
