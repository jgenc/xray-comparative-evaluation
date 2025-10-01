# TODO: Make this another file to just import the dataset
_base_ = [
    'co_dino_5scale_r50_1x_coco.py'
]

# What is happening here:
# - Backbone pre-trained on ImageNet-22k
# - Method is trained using the Co-DETR training scheme (which can be found by 
#   tracing the base dependencies)
# - The model is trained for 1 epoch (for now)
# - Training on X-ray Dataset
#
# Interestingly, the ViT backbone method (which I am not using here) is handled
# in a very different way:
# - Initialization using EVA-02
# - Intermediate fine-tuning on Objects365 for 26 epochs and reducing lr by 0.1
#   at epoch 24
# - Finally, the whole model is fine-tuned on COCO for 12 epochs, with learning
#   rates specified in the appendix
#
# The reason why I am noting this down is because _i am not using ViT, I am
# using SWIN and I am NOT following the ViT training/fine-tuning scheme_.

num_classes = 5
dataset_type = "CocoDataset"
data_root = "/home/cv-group/jorgen/ceasefire/ceasefire/Datasets/SIXray/Mode-10X-COCO"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file=data_root + '/annotations/train.json',
        img_prefix=data_root + '/images/train',
        classes = ("Gun", "Knife", "Wrench", "Pliers", "Scissors", )),
    val=dict(
        ann_file=data_root + '/annotations/val.json',
        img_prefix=data_root + '/images/val',
        classes = ("Gun", "Knife", "Wrench", "Pliers", "Scissors", )),
    test=dict(
        ann_file=data_root + '/annotations/test.json',
        img_prefix=data_root + '/images/test',
        classes = ("Gun", "Knife", "Wrench", "Pliers", "Scissors", ))
)


# pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth"
# models directory exists in ./runs (from root)
pretrained = "runs/models/swin_base_patch4_window12_384_22k.pth"

# model settings
num_dec_layer = 6
lambda_2 = 2.0

model = dict(
    # THIS IS NOT MMDETECTION SWIN, THIS IS CO-DETR SWIN!!! (WHICH IS MICROSOFT'S model)
    backbone=dict(
        frozen_stages=-1,
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=128, # 192 is the Swin-l version
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], # [6, 12, 24, 48] is the Swin-l version
        out_indices=(0, 1, 2, 3),
        window_size=12, # The Swin paper uses window_size=7, not 12. Why are the Co-DETR authors using 12?
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        pretrained=pretrained,
        # pretrain_img_size=224,
    ),
    neck=dict(in_channels=[128, 128*2, 128*4, 128*8]), # 192, 192*2, 192*4, 192*8 is the Swin-l version
    query_head=dict(
        num_classes=num_classes,
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6
            )
        )
    ),
    # Note: Copying both heads directly from the original config because
    # the original config uses a different number of classes
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0*num_dec_layer*lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0*num_dec_layer*lambda_2)))
    ],
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0*num_dec_layer*lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2)),],
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

# Taken directly from `co_dino_5scale_r50_1x_coco.py`
optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2),
    # cumulative_iters=0,
    # type="GradientCumulativeOptimizerHook",
)

model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(save_best="bbox_mAP_50")