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

## Datasets
* Factory [[dataset](https://zenodo.org/records/7035788)]
* Corridors [[dataset](https://zenodo.org/records/7035788)]

Download the datasets into ``dataset`` folder, like ``./dataset/Factory/testing/frames/01/``

## Model
* Our model can be obtained at the following link: ([Google Drive]https://drive.google.com/drive/folders/1NEdYzWAVl4IauO-SWxxtBJQHOW_SkApE?usp=sharing)
* or([Baidu Cloud]https://pan.baidu.com/s/17JyZae8RnMUPxq4jv3xRbg?pwd=49bw)

## Environment Configuration
```bash
conda create -n mnad python=3.9 -y
conda activate mnad

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

pip install -r requirements.txt
```
## Dataset structure
```bash
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

## Train
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
* Check your dataset_type (Factory or Corridors)
* Test the model with our pre-trained model and memory items
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

