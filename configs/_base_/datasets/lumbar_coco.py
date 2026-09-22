"""MMPose metainfo for the 22-point LumbarCoco landmark layout."""


KEYPOINT_NAMES = [
    *(f'L{level}{plate}{side}' for level in range(1, 6)
      for plate in ('T', 'B') for side in ('F', 'B')),
    'S1TF',
    'S1TB',
]


# Keypoints (22)
keypoint_info = {
    0: dict(
        name='L1TF',
        id=0,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    1: dict(
        name='L1TB',
        id=1,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    2: dict(
        name='L1BF',
        id=2,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    3: dict(
        name='L1BB',
        id=3,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    4: dict(
        name='L2TF',
        id=4,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    5: dict(
        name='L2TB',
        id=5,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    6: dict(
        name='L2BF',
        id=6,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    7: dict(
        name='L2BB',
        id=7,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    8: dict(
        name='L3TF',
        id=8,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    9: dict(
        name='L3TB',
        id=9,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    10: dict(
        name='L3BF',
        id=10,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    11: dict(
        name='L3BB',
        id=11,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    12: dict(
        name='L4TF',
        id=12,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    13: dict(
        name='L4TB',
        id=13,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    14: dict(
        name='L4BF',
        id=14,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    15: dict(
        name='L4BB',
        id=15,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    16: dict(
        name='L5TF',
        id=16,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    17: dict(
        name='L5TB',
        id=17,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    18: dict(
        name='L5BF',
        id=18,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    19: dict(
        name='L5BB',
        id=19,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
    20: dict(
        name='S1TF',
        id=20,
        color=[255, 128, 0],
        type='lower',
        swap='',
    ),
    21: dict(
        name='S1TB',
        id=21,
        color=[0, 255, 0],
        type='lower',
        swap='',
    ),
}

skeleton_info = {}
link_id = 0
for level in range(1, 6):
    links = (
        (f'L{level}TF', f'L{level}TB'),
        (f'L{level}BF', f'L{level}BB'),
        (f'L{level}TF', f'L{level}BF'),
        (f'L{level}TB', f'L{level}BB'),
    )
    for link in links:
        skeleton_info[link_id] = dict(
            link=link, id=link_id, color=[51, 153, 255])
        link_id += 1
skeleton_info[link_id] = dict(
    link=('S1TF', 'S1TB'), id=link_id, color=[255, 128, 0])

dataset_info = dict(
    dataset_name='LumbarCoco',
    paper_info=dict(
        author='Burapha University et al.',
        title='BUU-LSPINE: A Thai Open Lumbar Spine Dataset for '
              'Spondylolisthesis Detection',
        container='Applied Sciences',
        year='2023',
        homepage='https://services.informatics.buu.ac.th/spine/'),
    keypoint_info=keypoint_info,
    skeleton_info=skeleton_info,
    joint_weights=[1.0] * len(KEYPOINT_NAMES),
    # Uniform initial OKS tolerances. These should be re-estimated from repeat
    # annotations if independent annotations become available.
    sigmas=[0.05] * len(KEYPOINT_NAMES),
)
