# LumbarCoco / RTMPose-m

Актуальная пошаговая инструкция по обучению, TensorBoard и визуализации
предсказаний находится в [`RUN_TRAINING.md`](RUN_TRAINING.md).

This project converts the 400 lateral (`LA`) BUU-LSPINE_400 radiographs to a
22-keypoint COCO dataset and trains a one-stage RTMPose-m model.

## Landmark mapping

Every BUU CSV row is one endplate: `x1,y1,x2,y2,class`. BUU normalizes all LA
images so the patient faces image-left; therefore the endpoint with smaller x
is front/anterior and the endpoint with larger x is back/posterior. The
converter deliberately sorts the pair: 23 rows in three source files have the
opposite endpoint order. Rows map as follows:

1. L1 top -> `L1TF`, `L1TB`
2. L1 bottom -> `L1BF`, `L1BB`
3. The same top/bottom order for L2, L3, L4 and L5
4. S1 top -> `S1TF`, `S1TB`

The fifth CSV value is a spondylolisthesis class attached to selected lower
endplates. It is intentionally excluded because this task is landmark pose
estimation, not diagnosis classification.

## Build the dataset

From this repository root, run:

```bash
python tools/convert_buu_to_lumbar_coco.py
```

The deterministic seed-42 split is 320 train / 40 validation / 40 test. The
default creates `data/LumbarCoco/images` as a symlink to the original `LA`
directory, so it does not duplicate roughly half of the 890 MB source data.
For a self-contained dataset instead use:

```bash
python tools/convert_buu_to_lumbar_coco.py --image-mode copy --overwrite
```

The output is:

```text
data/LumbarCoco/
├── images -> ../../BUU-LSPINE_400/LA
└── annotations/
    ├── lumbar_keypoints_train.json
    ├── lumbar_keypoints_val.json
    └── lumbar_keypoints_test.json
```

Each image has one `lumbar_spine` COCO instance. All 22 points have visibility
2. Its bounding box is the landmark extent plus 10% padding. COCO keypoint AP
uses the 22 OKS sigmas in the MMPose metainfo; their initial value is 0.05 and
should be estimated from repeated expert annotations if those become
available.

## Train and evaluate

The project already has MMPose 1.3.2 in the `mmpose` Conda environment. Run
from this repository root so both `data/` and `configs/` resolve correctly:

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  --work-dir work_dirs/rtmpose-m_lumbar-coco \
  --amp
```

Alternatively activate the environment first with `conda activate mmpose` and
use the same command beginning with `python tools/train.py`. Do not use the
system `/usr/bin/python`: it is Python 3.14 and has no MMPose installation.
The `--no-capture-output` flag is important: without it, `conda run` buffers
the training log and the terminal can remain blank until the process exits.

For testing, replace the checkpoint path below with the generated best file:

```bash
conda run --no-capture-output -n mmpose python tools/test.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  work_dirs/rtmpose-m_lumbar-coco/best_PCK_epoch_*.pth
```

To overlay the ground-truth landmarks and model prediction on one validation
image, run:

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  work_dirs/rtmpose-m_lumbar-coco/best_PCK_epoch_110.pth \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --show-labels
```

Without an image argument the script uses the first image from the annotation
file. To process a specific image, add its path after the checkpoint. The
generated overlay is written to `output/comparisons/`. Ground-truth landmarks
are green circles, predictions are magenta crosses, and pale lines connect
each prediction to its target. The header also reports mean pixel error and
bbox-normalized PCK@0.05.

The model is RTMPose-m with 22 output channels and `KLDiscretLoss`. AdamW is
used with learning rate `5e-4` and weight decay `0.05`. There is one pipeline
for all 210 epochs—no stage-2 pipeline or `PipelineSwitchHook`. Validation
reports COCO AP and bbox-normalized PCK at thresholds 0.05, 0.03, 0.02 and
0.01; checkpoints are ranked by `PCK@0.05/PCK`.

Horizontal flip and human-specific half-body augmentation are disabled because
front/back are anatomical labels and every BUU LA image has the same facing
direction. The pose crop is 256x512 to better match the tall lumbar region.
