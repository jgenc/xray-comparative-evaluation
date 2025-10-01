# xray-comparative-evaluation

## Ultralytics

Original commit tree: https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3

The changes that have been applied to this specific version of ultralytics are tracked in the ultralytics [CHANGELOG](./ultralytics/CHANGES.md).

### Installation

0. Requirements:

  * Python 3.11.9
  * Package requirements as specified by this specific version of [ultralytics](https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3), or [`requires.txt`](./ultralytics/ultralytics.egg-info/requires.txt)

1. Create a conda environment:

```bash
cd ultralytics
conda create -n xray_ultralytics python=3.11 -y
conda activate xray_ultralytics
```

2. Install required packages

```bash
pip install -e .
```

### Usage

Read the provided [scripts file](./ultralytics/scripts.md) on how to train and validate the ultralytics-based models.

## dino

Original commit tree: https://github.com/open-mmlab/mmdetection/tree/fe3f809a0a514189baf889aa358c498d51ee36cd

The changes that have been applied to this specific version of mmdetection are tracked in the mmdetection [CHANGELOG](./dino/mmdetection/CHANGES.md).

### Installation

Follow the official installation guide from mmdetection [here](https://mmdetection.readthedocs.io/en/latest/get_started.html). Alternatively, follow the guide below.

0. Requirements:
* Python 

1. Create a conda environment and activate it:

```bash
conda create -n xray_dino python=3.11 -y
conda activate xray_dino
```

2. Install required packages

```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install mmengine and mmcv

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```

4. Install mmdet

```bash
cd mmdetection
pip install -e .
```

5. Install correct numpy version

```bash
pip install numpy==1.16.2
```

### Usage

Read the provided [scripts file](./dino/mmdetection/scripts.md) on how to train and validate the DINO model.

## Co-DETR

Original repository: https://github.com/Sense-X/Co-DETR

No changes have accurred in this repository, apart from new configuration files, which are provided at `co-detr/cfg`.

> [!NOTE]
> The mmdetection version of dino includes experimental code for CO-DETR in the projects folder, but we opted for the original implementation.

### Installation

1. Create conda environment using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate Co-DETR
```

2. Install mmdet

```bash
cd mmdetection
pip install -e .
```

3. Download the Swin backbone weights

```bash
wget "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth" -O swin_base_patch4_window12_384_22k.pth
```

### Usage

Read the provided [scripts file](./co-detr/mmdetection/scripts.md) on how to train and validate the Co-DETR model.

## Model weights

The pre-trained model weights for all models used in this comparative evaluation are available on Hugging Face: [jgen/xray-comparative-evaluation](https://huggingface.co/jgen/xray-comparative-evaluation)

**Download all weights:**
```bash
./scripts/download_weights.sh
```

**Manual Download:**
You can also download specific model weights directly from the Hugging Face repository or use the Hugging Face CLI:
```bash
# Example for YOLOv8 weights trained on CLCXray 
huggingface-cli download "jgen/xray-comparative-evaluation" "yolov8/yolov8_CLCXray/best.pt" --local-dir "weights" --local-dir-use-symlinks False
```

This will download the selected model weights to the `weights/` directory, organized by model architecture and dataset.