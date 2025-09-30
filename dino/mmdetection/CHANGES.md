# Changelog

Changes:
* Changed `classwise: bool = False` to `classwise: bool = False`, due to a bug that ignored this variable change in the model configurations. Important to have this set to True to obtain class-wise results.

Additions:
* `train_big.py` and `dist_train_big.py`, which contains an import to `FlexibleRunner` and uses that instead of the default `Runner`. Used in experiments where OOM errors frequently occurred (such as PIDray).

Comments:
* The COCO format for datasets lists `area` as an optional attribute, however this version of mmdetection directly accesses the attribute, which errors out. **Make sure all bbox entries contain the `area` attribute!**