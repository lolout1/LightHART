# LightHART: Lightweight Human Activity Recognition Transformer
Implementation of "Lightweight Human Activity Recognition Transformer" accepted in ICPR 2024.

## Newer or alternate versions of LightHART and related works 

- **FusionTransformer (Abheek Pradhan's current work is located here ( Fall 2025 )**  
  *Current repository focusing on multimodal fusion:*  
  https://github.com/lolout1/FusionTransformer
  
- **LightHART (TensorFlow version - supporting compatible ops for conversion to ONNX or TensorflowLite ):**  
  https://github.com/lolout1/LightHART-tf

- **FeatureKD (Tousiful Haque)**  
  *Previously, LightHART was a fork of Tousiful’s repository (formerly also named LightHART before the rename):*  
  https://github.com/tousifulhaque/FeatureKD



## Getting started 
- Create an pip environment and use the requirements.txt to install all the neccasary files.

```bash
pip install -r requirements.txt
```
This requirements file doesn't have the instructions to install pytorch. Please install pytorch 1.13.0 for the experiments

## Get the Dataset
- Download the SmartFallMM data from [this link](https://github.com/tousifulhaque/smartfallmm.git). This is a a private repo so, please ask me to add you as a collaborator to access the dataset. Put the dataset under `data` folder. 


## Choose and configure models
- Model configuration for Accelerometer model is kept under ``config/smartfallmm/student.yaml``.

- Model Configuration or Skeleton model is kept under ``config/smartfallmm/teacher.yaml`` for SmartFallMM dataset.

- Configuration for Distillation is stored in ``config/smartfallmm/distill.yaml`` for SmartFallMM dataset.

## Step for Knowledge Distillation 
- Train a teacher model first using ``main.py`` and ``config/smartfallmm/teacher.yaml``. You can do it by uncommenting `line 38` in `train.sh` . 
- Perform knowledge distillation using ``distill.py`` and ``config/smartfallmm/distill.yaml`` . You can perform the distillation by uncomenting `line 46`.

## Train and test
Give execution access to ``train.sh`` with 
```bash
chmod +x ./train.sh
```
Run the ``train.sh`` to train and test the multimodal and accelerometer models. Log and weights would be saved under working directory. Use the following command to run the ``train.sh`` script.

```bash
./train.sh
```


