"""BUU-2000 RTMPose-m with asymmetric bbox augmentation and Local PCK."""

_base_ = ['./rtmpose-m_lumbar-coco2000_1stage.py']

custom_imports = dict(imports=['augmentations.bbox', 'metric.local_pck'],
                      allow_failed_imports=False)

# Fractions of the original annotation bbox width (left/right) or height
# (top/bottom). Each side is selected independently when augmentation applies.
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(
        type='RandomLumbarBBoxExpansion',
        apply_prob=0.8,
        clip_to_image=True,
        side_probs=dict(left=0.5, right=0.5, top=0.5, bottom=0.5),
        expansion_ranges=dict(
            left=(0.0, 0.5),
            right=(0.0, 0.5),
            top=(0.0, 0.5),
            bottom=(0.0, 0.5),
        ),
        rotation_range=(-15.0, 15.0),
    ),
    dict(type='TopdownAffine', input_size=(256, 512)),
    dict(type='GenerateTarget', encoder=_base_.codec),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_evaluator = [
    dict(type='LocalPCK', threshold=0.1, mode='segment',
         prefix='LocalPCK/segment@0.1'),
]
test_evaluator = [
    dict(type='LocalPCK', threshold=0.1, mode='segment',
         prefix='LocalPCK/segment@0.1'),
]

default_hooks = dict(
    checkpoint=dict(save_best='LocalPCK/segment@0.1/PCK', rule='greater'))
