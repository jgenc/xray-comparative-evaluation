# Illicit object detection in X-ray imaging using deep learning techniques: A comparative evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2507.17508-b31b1b.svg)](https://arxiv.org/abs/2507.17508)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/jgen/xray-comparative-evaluation)

This repository contains the official code and pre-trained models for the paper "Illicit object detection in X-ray imaging using deep learning techniques: A comparative evaluation". In this work, we conduct a systematic and thorough comparative evaluation of recent Deep Learning (DL)-based methods for X-ray object detection. We evaluate ten state-of-the-art object detectors across six major public datasets to provide critical insights into their performance, efficiency, and real-world applicability

## Key Features

  * **10 State-of-the-Art Models**: Implementations for generic CNNs (YOLOv8), custom CNNs (CHR, DOAM, LIM), transformers (DINO, Co-DETR), and hybrid architectures (RT-DETR, Next-ViT).
  * **6 Public Datasets**: A unified framework to train and evaluate on OPIXray, CLCXray, SIXray, EDS, HiXray, and PIDray.
  * **Reproducible Framework**: Detailed instructions, environment setups, and training/validation scripts to ensure full reproducibility of our results.
  * **Pre-trained Weights**: All model weights are available on Hugging Face for immediate use and validation.

## Main results

Object detection performance is reported as $mAP^{50}$ / $mAP^{50:95}$. Model configurations are denoted as D(detection_head, backbone_network).

| Configuration | OPIXray | CLCXray | SIXray | EDS (avg.) | HIXray | PIDray (overall) | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Generic CNN detectors** | | | | | | | |
| D(YOLOv8, CSPDarkNet53) | 0.868 / 0.413 | 0.733 / 0.636 | 0.901 / **0.794** | 0.547 / 0.386 | 0.845 / **0.564** | 0.897 / **0.807** | 0.799 / 0.600 |
| D(YOLOv8, HGNetV2) | 0.898 / 0.418 | 0.725 / 0.617 | 0.897 / 0.775 | 0.550 / 0.378 | 0.833 / 0.557 | 0.902 / 0.796 | 0.801 / 0.590 |
| **Custom CNN detectors** | | | | | | | |
| D(YOLOv8+CHR, CSPDarkNet53) | 0.835 / 0.368 | 0.710 / 0.602 | 0.850 / 0.700 | 0.416 / 0.276 | 0.811 / 0.523 | 0.782 / 0.644 | 0.734 / 0.519 |
| D(YOLOv8+DOAM, CSPDarkNet53)| 0.790 / 0.361 | 0.720 / 0.614 | 0.828 / 0.658 | 0.422 / 0.280 | 0.830 / 0.545 | 0.815 / 0.689 | 0.734 / 0.525 |
| D(YOLOv8+LIM, CSPDarkNet53) | 0.791 / 0.344 | 0.717 / 0.605 | 0.827 / 0.661 | 0.446 / 0.300 | 0.828 / 0.525 | 0.800 / 0.664 | 0.735 / 0.517 |
| **Generic transformer detectors** | | | | | | | |
| D(DINO, Swin-B) | 0.928 / 0.413 | 0.739 / 0.607 | 0.902 / 0.765 | 0.560 / 0.378 | 0.849 / 0.535 | 0.802 / 0.655 | 0.797 / 0.559 |
| D(Co-DETR, Swin-B) | 0.928 / 0.423 | 0.772 / **0.654** | 0.893 / 0.735 | 0.653 / **0.450** | 0.857 / 0.531 | 0.852 / 0.732 | 0.826 / 0.587 |
| **Generic hybrid detectors** | | | | | | | |
| D(RT-DETR, HGNetV2) | 0.898 / 0.389 | 0.721 / 0.609 | 0.901 / 0.789 | 0.573 / 0.410 | 0.839 / 0.510 | 0.835 / 0.720 | 0.795 / 0.571 |
| D(YOLOv8, Next-ViT-S) | 0.906 / **0.429** | 0.740 / 0.640 | 0.906 / 0.793 | 0.588 / 0.408 | 0.841 / 0.551 | 0.898 / 0.801 | 0.813 / **0.604** |
| D(RT-DETR, Next-ViT-S) | 0.887 / 0.389 | 0.720 / 0.609 | 0.889 / 0.762 | 0.504 / 0.322 | 0.818 / 0.483 | 0.879 / 0.773 | 0.783 / 0.556 |

## Repository Structure

```
xray-comparative-evaluation/
├── ultralytics/
│   ├── scripts.md          # Training and validation scripts
│   └── CHANGES.md          # Modifications log
├── dino/
│   └── mmdetection/
│       ├── scripts.md
│       └── CHANGES.md
├── co-detr/
│   ├── mmdetection/
│   │   └── scripts.md
│   └── cfg/                # Configuration files
├── scripts/                # Utility scripts
│   └── download_weights.sh
└── README.md
```

## Environment preparation

### Ultralytics

Original commit tree: https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3

The changes that have been applied to this specific version of ultralytics are tracked in the ultralytics [CHANGELOG](./ultralytics/CHANGES.md).

#### Installation

> [!NOTE]
> Package requirements as specified by this specific version of [ultralytics](https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3), or [`requires.txt`](./ultralytics/ultralytics.egg-info/requires.txt)

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

#### Usage

Read the provided [scripts file](./ultralytics/scripts.md) on how to train and validate the ultralytics-based models.

---

### DINO

Original commit tree: https://github.com/open-mmlab/mmdetection/tree/fe3f809a0a514189baf889aa358c498d51ee36cd

The changes that have been applied to this specific version of mmdetection are tracked in the mmdetection [CHANGELOG](./dino/mmdetection/CHANGES.md).

#### Installation

Follow the official installation guide from mmdetection [here](https://mmdetection.readthedocs.io/en/latest/get_started.html). Alternatively, follow the guide below.

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

#### Usage

Read the provided [scripts file](./dino/mmdetection/scripts.md) on how to train and validate the DINO model.

---

### Co-DETR

Original repository: https://github.com/Sense-X/Co-DETR

No changes have accurred in this repository, apart from new configuration files, which are provided at `co-detr/cfg`.

> [!NOTE]
> The mmdetection version of dino includes experimental code for CO-DETR in the projects folder, but we opted for the original implementation.

#### Installation

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

#### Usage

Read the provided [scripts file](./co-detr/mmdetection/scripts.md) on how to train and validate the Co-DETR model.

### Model weights

The pre-trained model weights for all models used in this comparative evaluation are available on Hugging Face: [jgen/xray-comparative-evaluation](https://huggingface.co/jgen/xray-comparative-evaluation).

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

## Dataset preparation

> [!CAUTION]
> Correct data preparation is crucial for the scripts to work as intended.

This work evaluates models on six public datasets: **OPIXray, CLCXray, SIXray, EDS, HiXray, and PIDray**. You must download them from their official sources.

### Annotation Formats

The frameworks used in this repository require different annotation formats:

1. **COCO Format**: Required for **DINO** and **Co-DETR** (MMDetection-based).
2. **YOLO Format**: Required for all **Ultralytics**-based models.

> [!TIP]
> We recommend converting all datasets to **COCO format** first. You can then easily convert from COCO to YOLO format using Ultralytics' [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool.

#### Recommended Directory Structure

To ensure the scripts work out-of-the-box, we suggest organizing your datasets as follows:

```
datasets/
├── OPIXray/
│   ├── annotations/    # Used in the COCO format
│   ├── labels/         # Used in the YOLO format
│   └── images/
├── CLCXray/
│   ├── annotations/
│   ├── labels/
│   └── images/
└── ... (and so on for other datasets)
```

## Citation

If you are using this repository or our research findings in your own research, please cite using the following:

```latex
@article{cani2025illicit,
  title={Illicit object detection in X-ray imaging using deep learning techniques: A comparative evaluation},
  author={Cani, Jorgen and Diou, Christos and Evangelatos, Spyridon and Argyriou, Vasileios and Radoglou-Grammatikis, Panagiotis and Sarigiannidis, Panagiotis and Varlamis, Iraklis and Papadopoulos, Georgios Th},
  journal={arXiv preprint arXiv:2507.17508},
  year={2025}
}
```

## Acknowledgments

This work wouldn't be possible without the foundational open-source projects:

  - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
  - [MMDetection](https://github.com/open-mmlab/mmdetection)
  - [Co-DETR](https://github.com/Sense-X/Co-DETR)
  - [Next-ViT](https://github.com/bytedance/Next-ViT)

We also extend our gratitude to the creators of the OPIXray, CLCXray, SIXray, EDS, HiXray, and PIDray datasets.