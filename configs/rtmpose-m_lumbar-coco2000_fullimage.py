"""Controlled baseline: full source image, no random geometric augmentation."""
_base_ = ['./rtmpose-m_lumbar-coco2000_bbox-localpck.py']
custom_imports = dict(imports=['augmentations.vertebrae_crop', 'metric.local_pck'],
                      allow_failed_imports=False)
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='FullImageBBox'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=(256, 512)),
    dict(type='GenerateTarget', encoder=_base_.codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='FullImageBBox'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=(256, 512)),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
