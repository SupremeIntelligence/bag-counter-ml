_base_ = "../rtmdet_tiny_8xb32-300e_coco.py"

data_root = "data/dataset_v2/"

metainfo = {
    "classes": ("bag",),
}


model = dict(
    bbox_head=dict(
        num_classes=1,
    )
)


train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/annotations.json",
        data_prefix=dict(img="train/images/"),
        filter_cfg=dict(
            filter_empty_gt=False,
            min_size=32,
        ),
    ),
)


val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annotations.json",
        data_prefix=dict(img="val/images/"),
    ),
)

test_dataloader = val_dataloader


val_evaluator = dict(
    ann_file=data_root + "val/annotations.json",
)

test_evaluator = val_evaluator


#Fine-tuning from COCO-pretrained RTMDet-tiny
load_from = (
    "rtmdet_tiny_8xb32-300e_coco_"
    "20220902_112414-78e30dcc.pth"
)


optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
    )
)


param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=100,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        begin=1,
        end=50,
        T_max=49,
        by_epoch=True,
    ),
]


train_cfg = dict(
    max_epochs=50,
    val_interval=1000,
)


custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    )
]


default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        max_keep_ckpts=10,
    )
)