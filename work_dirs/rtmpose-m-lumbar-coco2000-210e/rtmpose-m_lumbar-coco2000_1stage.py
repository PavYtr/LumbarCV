BATCH = 64
MAX_EPOCHS = 210
WORKERS = 8
auto_scale_lr = dict(base_batch_size=32)
backend_args = dict(backend='local')
base_lr = 0.0005
codec = dict(
    input_size=(
        256,
        512,
    ),
    normalize=False,
    sigma=(
        6.0,
        9.0,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_mode = 'topdown'
data_root = 'data/LumbarCoco2000/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        _scope_='mmpose',
        interval=10,
        max_keep_ckpts=3,
        rule='greater',
        save_best='PCK@0.05/PCK',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=10, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    256,
    512,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
metainfo = dict(from_file='configs/_base_/datasets/lumbar_coco.py')
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                256,
                512,
            ),
            normalize=False,
            sigma=(
                6.0,
                9.0,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            8,
            16,
        ),
        input_size=(
            256,
            512,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=22,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=False),
    type='TopdownPoseEstimator')
num_keypoints = 22
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=105,
        begin=105,
        by_epoch=True,
        convert_to_iter_based=True,
        end=210,
        eta_min=2.5e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/lumbar_keypoints_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/LumbarCoco2000/',
        metainfo=dict(from_file='configs/_base_/datasets/lumbar_coco.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.15, type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                512,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='data/LumbarCoco2000/annotations/lumbar_keypoints_test.json',
        type='CocoMetric'),
    dict(norm_item='bbox', prefix='PCK@0.05', thr=0.05, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.03', thr=0.03, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.02', thr=0.02, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.01', thr=0.01, type='PCKAccuracy'),
]
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/lumbar_keypoints_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/LumbarCoco2000/',
        metainfo=dict(from_file='configs/_base_/datasets/lumbar_coco.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.15, type='GetBBoxCenterScale'),
            dict(
                rotate_factor=15,
                scale_factor=[
                    0.85,
                    1.15,
                ],
                shift_factor=0.1,
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                512,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        256,
                        512,
                    ),
                    normalize=False,
                    sigma=(
                        6.0,
                        9.0,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(padding=1.15, type='GetBBoxCenterScale'),
    dict(
        rotate_factor=15,
        scale_factor=[
            0.85,
            1.15,
        ],
        shift_factor=0.1,
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        512,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                256,
                512,
            ),
            normalize=False,
            sigma=(
                6.0,
                9.0,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/lumbar_keypoints_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/LumbarCoco2000/',
        metainfo=dict(from_file='configs/_base_/datasets/lumbar_coco.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.15, type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                512,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file='data/LumbarCoco2000/annotations/lumbar_keypoints_val.json',
        type='CocoMetric'),
    dict(norm_item='bbox', prefix='PCK@0.05', thr=0.05, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.03', thr=0.03, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.02', thr=0.02, type='PCKAccuracy'),
    dict(norm_item='bbox', prefix='PCK@0.01', thr=0.01, type='PCKAccuracy'),
]
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(padding=1.15, type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        512,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs/rtmpose-m-lumbar-coco2000-210e'
