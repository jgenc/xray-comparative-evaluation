# Changelog

Changes:
* `cfg` folder contains configuration files that were used for training. Specifically:
  * `cfg/datasets/` contains `yaml` files of the used datasets in this work. The datasets must contain YOLO style annotations for these configurations to work. We have used the following tool to covert COCO-style annotations to YOLO: ... Some of the datasets used did not provide COCO-style annotations, meaning that we had to convert them ourselves to COCO, essentially doing the follwing: Other style -> COCO -> YOLO.
  * `cfg/models` contains the `yaml` files of the used models in this work.
* `scripts` contains training and testings scripts
* `error_analysis` (**subject to change**) contains files used for generating the coco error analysis figures.
* In `ultralytics/nn/tasks.py` we've modified the code to make use of the modules needed by Next-ViT-S, LIM, DOAM and CHR. Specfifically:
  * In `parse_model(...)`, we've added the modules needed for the aforementioned methods
  * In `yaml_model_load(...)`, because at that version of `ultralytics`, the scale parameter was almost always ignored.
* In `ultralytics/nn/modules/__init__.py` we've imported the additional modules needed by Next-ViT-S, LIM, DOAM, and CHR.
* In `ultralytics/nn/modules/block.py` we've imported the additional modules needed by LIM, DOAM, and CHR.

Additions:
* `ultralytics/nn/modules/nextvit_utils.py` contains code from the original Next-ViT repository, modified to work with ultralytics
* `ultralytics/nn/modules/nextvit.py` contains code from the original Next-ViT repository, modified to work with ultralytics

Comments:
* All RT-DETR based model weights **must** contain `rt_detr`, otherwise the results of their evaluation will not work!