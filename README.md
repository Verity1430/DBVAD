# Dual-Branch Collaborative Framework for Robust Visual Anomaly Detection in Mobile Robot Inspection
## Related manuscript

This repository contains the code, dependencies, and usage instructions associated with the manuscript currently submitted to *The Visual Computer*.

This code is directly related to the submitted manuscript and is provided to support transparency, reproducibility, and evaluation of the reported results.

If you use this repository, please cite this code release and, when available, the corresponding article in *The Visual Computer*.

## Dependencies
* Python 3.9.23
* PyTorch 2.2.2
* Cuda12.1
* numpy 1.24.4
* scikit-learn 1.3.2
* opencv-python-headless 4.8.1.78
* matplotlib 3.8.0
* tqdm 4.66.4
* fvcore 0.1.5.post20221221

## Environment Configuration
```bash
conda create -n mnad python=3.9 -y
conda activate mnad

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

pip install -r requirements.txt
```

## Datasets
This repository supports Factory and Corridors datasets.
* Factory [[dataset](https://zenodo.org/records/7035788)]
* Corridors [[dataset](https://zenodo.org/records/7035788)]

After obtaining the datasets, preprocess the data as image files (refer to below).
```bash
# Dataset preparation example:
dataset/
├── Factory/
│   ├── training/
│   │   └── frames/
│   │       └── 01/
│   └── testing/
│       └── frames/
│           └── 01/
└── Corridors/
    ├── training/
    │   └── frames/
    │       └── 01/
    └── testing/
        └── frames/
            └── 01/
```


## Pretrained Weights and Feature-Bank Resources

The following links provide the pretrained model weights and auxiliary resources required to reproduce our pipeline.
These resources include:
- pretrained model weights,
- the local visual backbone `vit_small_patch14_dinov2.lvd142m`.
* ([Google Drive]https://drive.google.com/drive/folders/1NEdYzWAVl4IauO-SWxxtBJQHOW_SkApE?usp=sharing)
* ([Baidu Cloud]https://pan.baidu.com/s/17JyZae8RnMUPxq4jv3xRbg?pwd=49bw)

## Train
This section describes how to train the prediction model and prepare the feature bank used in our framework.

Before running the commands below, please make sure that:
- the datasets have been organized under the `./dataset` directory as described above,
- all dependencies have been installed correctly,
- the selected GPU is available.

### 1. Train the model
First, train the prediction model on the target dataset.  
The trained checkpoints and logs will be saved in the directory specified by `--exp_dir`.
```bash
# Factory
python Train.py --gpus 0 \
  --dataset_type Factory \
  --dataset_path ./dataset \
  --exp_dir log \
  --epochs 60
```
```bash
# Factory
python Train.py --gpus 0 \
  --dataset_type Corridors \
  --dataset_path ./dataset \
  --exp_dir log \
  --epochs 60
```
### 2. Build the local feature bank
After training, run the following commands to generate the corresponding bank file for each dataset.
These files will be used later during evaluation and localization.
```bash
python Bank.py \
  --gpus 0 \
  --dataset_type Factory \
  --dataset_path ./dataset \
  --exp_dir log \
  --method pred \
  --local_backbone vit_small_patch14_dinov2.lvd142m \
  --local_max_items 50000 \
  --local_stride 10
```
```bash
python Bank.py \
  --gpus 0 \
  --dataset_type Corridors \
  --dataset_path ./dataset \
  --exp_dir log \
  --method pred \
  --local_backbone vit_small_patch14_dinov2.lvd142m \
  --local_max_items 50000 \
  --local_stride 10
```

## Evaluation
This section describes how to evaluate the trained model using the pretrained weights and the generated bank files.

Before evaluation, please check that:
- the dataset path is correct,
- the model checkpoint exists,
- the feature bank file exists,
- the output directory is writable.

The following commands will generate prediction results and save high-score frames for further inspection.
```bash
python Evaluate.py --gpus 0 \
  --dataset_type Factory \
  --dataset_path ./dataset \
  --model_dir ./exp/Factory/pred/log/model.pth \
  --m_items_dir ./exp/Factory/pred/log/keys.pt \
  --enable_localizer \
  --localizer_bank ./exp/Factory/pred/log/localizer_bank.pt \
  --local_backbone vit_small_patch14_dinov2.lvd142m \
  --save_high_frames \
  --max_save_per_video 6400 \
  --save_dir ./exp/Factory/pred/highframes
```
or
```bash
python Evaluate.py --gpus 0 \
  --dataset_type Corridors \
  --dataset_path ./dataset \
  --model_dir ./exp/Corridors/pred/log/model.pth \
  --m_items_dir ./exp/Corridors/pred/log/keys.pt \
  --enable_localizer \
  --localizer_bank ./exp/Corridors/pred/log/localizer_bank.pt \
  --local_backbone vit_small_patch14_dinov2.lvd142m \
  --save_high_frames \
  --max_save_per_video 20000 \
  --save_dir ./exp/Corridors/pred/highframes
```
After evaluation, the generated results can be used for qualitative inspection and further comparison with the reported results in the manuscript.
