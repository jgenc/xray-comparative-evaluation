# xray-comparative-evaluation

## Ultralytics

Original commit tree: https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3

### Installation

0. Requirements:

  * Python 3.11.9
  * Package requirements as specified by this specific version of [ultralytics](https://github.com/ultralytics/ultralytics/tree/765b98f44eb662c012e3470546bf95ea39345ee3), or [`requires.txt`](./ultralytics/ultralytics.egg-info/requires.txt)

1. Create a conda environment:

```bash
conda create env -n xray_ultralytics python=3.11 -y
```

2. Install required packages

```bash
pip install -e .
```

### Usage

Read the provided [scripts file](./ultralytics/scripts.md) on how to train and validate the ultralytics-based models.

The changes that have been applied to this specific version of ultralytics are tracked in the ultralytics [CHANGELOG](./ultralytics/CHANGES.md).

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