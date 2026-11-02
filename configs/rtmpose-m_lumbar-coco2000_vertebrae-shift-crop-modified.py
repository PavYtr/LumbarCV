"""Standalone RTMPose-m / LumbarCoco2000: full bbox, shift and spine crop.

Run from the repository root. Requires the local augmentations/metric modules,
LumbarCoco2000 data and reports/vertebrae_statistics_train.json.
"""

default_scope = 'mmpose'
custom_imports = dict(
    imports=['augmentations.vertebrae_shift', 'augmentations.vertebrae_crop',
             'metric.local_pck'],
    allow_failed_imports=False)

# Training and optimization.
MAX_EPOCHS = 210
BATCH = 64
WORKERS = 8
base_lr = 5e-4
train_cfg = dict(by_epoch=True, max_epochs=MAX_EPOCHS, val_interval=10)
val_cfg = dict()
test_cfg = dict()
randomness = dict(seed=21)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=MAX_EPOCHS // 2, end=MAX_EPOCHS, T_max=MAX_EPOCHS // 2,
         by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=32)

# Model and SimCC targets. input_size is (width, height).
num_keypoints = 22
input_size = (256, 512)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6.0, 9.0),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

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
        input_size=input_size,
        in_featuremap_size=tuple(size // 32 for size in input_size),
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
        loss=dict(type='KLDiscretLoss', use_target_weight=True,
                  beta=10.0, label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

# Dataset metadata is embedded so no other config files are needed.
keypoint_names = [
    *(f'L{level}{plate}{side}' for level in range(1, 6)
      for plate in ('T', 'B') for side in ('F', 'B')),
    'S1TF', 'S1TB',
]
skeleton_links = [
    link for level in range(1, 6) for link in (
        (f'L{level}TF', f'L{level}TB'),
        (f'L{level}BF', f'L{level}BB'),
        (f'L{level}TF', f'L{level}BF'),
        (f'L{level}TB', f'L{level}BB'),
    )
] + [('S1TF', 'S1TB')]
metainfo = dict(
    dataset_name='LumbarCoco',
    paper_info=dict(
        author='Burapha University et al.',
        title='BUU-LSPINE: A Thai Open Lumbar Spine Dataset for '
              'Spondylolisthesis Detection',
        container='Applied Sciences',
        year='2023',
        homepage='https://services.informatics.buu.ac.th/spine/'),
    keypoint_info={
        index: dict(name=name, id=index, type='lower', swap='',
                    color=[255, 128, 0] if name.endswith('F') else [0, 255, 0])
        for index, name in enumerate(keypoint_names)
    },
    skeleton_info={
        index: dict(link=link, id=index,
                    color=[255, 128, 0] if index == 20 else [51, 153, 255])
        for index, link in enumerate(skeleton_links)
    },
    joint_weights=[1.0] * num_keypoints,
    sigmas=[0.05] * num_keypoints)

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/LumbarCoco2000/'
backend_args = dict(backend='local')

# Shift first; the crop then uses the updated landmarks. FullImageBBox always
# sets bbox to [0, 0, width, height]. Crop changes center/scale/rotation only.
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='RandomVertebraeShift', prob=0.2, max_shift_ratio=1.25),
    dict(type='FullImageBBox'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='RandomVertebraeCrop',
         statistics_file='reports/vertebrae_statistics_train.json',
         apply_prob=0.8),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='FullImageBBox'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=BATCH,
    num_workers=WORKERS,
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

val_evaluator = [
    dict(type='LocalPCK', threshold=0.1, mode='segment',
         prefix='LocalPCK/segment@0.1'),
]
test_evaluator = [
    dict(type='LocalPCK', threshold=0.1, mode='segment',
         prefix='LocalPCK/segment@0.1'),
]

# Runtime, checkpoints and EMA.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=10,
        save_best='LocalPCK/segment@0.1/PCK', rule='greater', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(type='BadCaseAnalysisHook', enable=False, out_dir='badcase',
                 metric_type='loss', badcase_thr=5))
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
