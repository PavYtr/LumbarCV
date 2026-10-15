"""Same training as fullimage baseline, with empirical rigid spine crop."""
_base_ = ['./rtmpose-m_lumbar-coco2000_fullimage.py']
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='FullImageBBox'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    # JSON mean.margins_relative -> expansion_ranges=(0, mean) per side;
    # mean.absolute_angle_deg -> rotation_range=(-mean_abs, +mean_abs).
    dict(type='RandomVertebraeCrop',
         statistics_file='reports/vertebrae_statistics_train.json', apply_prob=0.8),
    dict(type='TopdownAffine', input_size=(256, 512)),
    dict(type='GenerateTarget', encoder=_base_.codec),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
