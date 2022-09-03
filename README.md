# C2PDN

Due to the limitations of the github anonymous repository, it is not possible to directly clone the entire repository

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
