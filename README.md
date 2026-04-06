# The complete code will be publicly released after the manuscript is accepted. Below is our training model and test code for your reference.

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
* Our model can be obtained at the following link: ([Google Drive]https://drive.google.com/drive/folders/1NEdYzWAVl4IauO-SWxxtBJQHOW_SkApE?usp=sharing)or([Baidu Cloud]https://pan.baidu.com/s/17JyZae8RnMUPxq4jv3xRbg?pwd=49bw)

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

