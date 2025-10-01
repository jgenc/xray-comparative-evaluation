configs = [
    "co_dino_5scale_swin_base_3x_coco_EDS_1_2.py",
    "co_dino_5scale_swin_base_3x_coco_EDS_1_3.py",
    "co_dino_5scale_swin_base_3x_coco_EDS_2_1.py",
    "co_dino_5scale_swin_base_3x_coco_EDS_2_3.py",
    "co_dino_5scale_swin_base_3x_coco_EDS_3_1.py",
    "co_dino_5scale_swin_base_3x_coco_EDS_3_2.py",
]


for config in configs:
    run_name = config.split("coco_")[1].split(".py")[0].lower()
    command = f"nohup ./tools/dist_train.sh ./projects/configs/co_dino/{config}.py 2 --word-dir runs/{run_name} & disown"
    print(command)
