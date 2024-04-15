This repo is the implementation of [MMTLB](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) with AVRET on CSL-Daily dataset. Thanks to their great work. 
It currently includes code and models for the gloss-based SLT task.

## Installation

```bash
conda env create -f environment.yml
conda activate slt
```

## Getting Started

### Preparation
Please refer to [MMTLB](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) for preparing the data and models.

The pretrained models of mBart_zh can download from [here](https://hkustconnect-my.sharepoint.com/personal/rzuo_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frzuo%5Fconnect%5Fust%5Fhk%2FDocuments%2Fpretrained%5Fmodels&ga=1). Please put it in the "pretrained_models" folder.
```shell
TwoStreamNetwork
└── pretrained_models
    └── mBart_zh
        └── ...
```
The pretrained s2t checkpoint of CSL-Daily SingleStream-SLT can download from [here](https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/SingleStream-SLT.md). Please put it in the "pretrained_models/csl-daily_s2t" folder.
```shell
TwoStreamNetwork
└── pretrained_models
    └── csl-daily_s2t
        └── ckpts
            └── best.ckpt
```
The pretrained S3D features of CSL-Daily can download from [here](https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/SingleStream-SLT.md). Please put it in the "experiments/outputs/SingleStream/extract_feature/head_rgb_input" folder.
```shell
TwoStreamNetwork
└── experiments
    └── outputs
        └── SingleStream
            └── extract_feature
               └── head_rgb_input
```

### Train
```bash
python training.py 
```

## Note
Since MMTLB is based on the MBart, we did not apply the local clip self-attention (LCSA) module to it.

To reduce the impact of gloss on SLT task, we only loaded the pretrained s2t checkpoint for visual_head and VLMapper module, and did not load it for TranslationNetwork.

# LICENSE
The code is released under the MIT license.