"""One-stage RTMPose-m training config for LumbarCoco (MMPose 1.x)."""

_base_ = ['mmpose::_base_/default_runtime.py']

# Runtime and optimization. The original RTMPose recipe's second pipeline and
# PipelineSwitchHook are intentionally omitted, leaving one training stage.
max_epochs = 420
base_lr = 5e-4
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False,
        begin=0, end=500),
    dict(
        type='CosineAnnealingLR', eta_min=base_lr * 0.05,
        begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2,
        by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=32)

# A taller crop preserves substantially more lumbar detail than the COCO
# human-pose 192x256 crop. input_size is (width, height).
num_keypoints = 22
input_size = (256, 512)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6.0, 9.0),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# TensorBoard visualiser
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-'
            'f2f7d6f6_20230130.pth')),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple(size // 32 for size in codec['input_size']),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=codec),
    # Front and back are distinct anatomical labels; horizontal flip testing
    # would require a right-facing training convention that this dataset lacks.
    test_cfg=dict(flip_test=False))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/LumbarCoco/'
metainfo = dict(from_file='configs/_base_/datasets/lumbar_coco.py')
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.10,
        scale_factor=[0.85, 1.15],
        rotate_factor=15),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/lumbar_keypoints_train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/lumbar_keypoints_val.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/lumbar_keypoints_test.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))

# PCK is the fraction of landmarks within 5% of the longer bbox side. Both AP
# and PCK are reported, while the checkpoint hook selects the highest PCK.
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/lumbar_keypoints_val.json'),
    dict(type='PCKAccuracy', thr=0.05, norm_item='bbox'),
]
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/lumbar_keypoints_test.json'),
    dict(type='PCKAccuracy', thr=0.05, norm_item='bbox'),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='PCK',
        rule='greater',
        max_keep_ckpts=3))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]
