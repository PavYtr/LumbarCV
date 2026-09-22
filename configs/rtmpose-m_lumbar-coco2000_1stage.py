"""RTMPose-m config for the 2,000-image LumbarCoco conversion."""

_base_ = ['./rtmpose-m_lumbar-coco_1stage.py']

data_root = 'data/LumbarCoco2000/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/lumbar_keypoints_train.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/lumbar_keypoints_val.json',
        data_prefix=dict(img='images/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/lumbar_keypoints_test.json',
        data_prefix=dict(img='images/')))

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/lumbar_keypoints_val.json'),
    dict(type='PCKAccuracy', thr=0.05, norm_item='bbox', prefix='PCK@0.05'),
    dict(type='PCKAccuracy', thr=0.03, norm_item='bbox', prefix='PCK@0.03'),
    dict(type='PCKAccuracy', thr=0.02, norm_item='bbox', prefix='PCK@0.02'),
    dict(type='PCKAccuracy', thr=0.01, norm_item='bbox', prefix='PCK@0.01'),
]
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/lumbar_keypoints_test.json'),
    dict(type='PCKAccuracy', thr=0.05, norm_item='bbox', prefix='PCK@0.05'),
    dict(type='PCKAccuracy', thr=0.03, norm_item='bbox', prefix='PCK@0.03'),
    dict(type='PCKAccuracy', thr=0.02, norm_item='bbox', prefix='PCK@0.02'),
    dict(type='PCKAccuracy', thr=0.01, norm_item='bbox', prefix='PCK@0.01'),
]
