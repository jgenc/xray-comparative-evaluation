# Scripts

## Training

```bash
yolo \
  task=detect \
  mode=train \
  model=/path/to/model/config/or/weights \
  data=/path/to/dataset/config \
  epochs=N \
  patience=N \
  batch=N \
  lr0=N \
  momentum=N \
  weight_decay=N
```

* `model` for training can either be: 
  * a model config (e.g., `.yaml` from `ultralytics/cfg/models/nextvit_rtdetr-l.yaml`),
  * or pre-trained model weights (e.g., `weights/nextvit_yolov8/nextvit_yolov8_CLCXray/best.pt`).
* `data` should always point to a valid dataset configuration file. Make sure to update `train`, `val` and `test` paths to the correct ones.
* The hyperparameters can take any value N (float or integer, depending on the parameter). Check https://docs.ultralytics.com/usage/cfg/ for more details.

## Testing

```bash
yolo \
  task=detect \
  mode=val \
  model=/path/to/trained/model/weights \
  data=/path/to/dataset/config \
  split=test
```

* `model` for evaluating can only be model weights!
* `split` is used to make sure that the testing will evaluate the specific set (i.e., `test`)