# Запуск обучения и визуализаций RTMPose-m / LumbarCoco

Все команды ниже выполняются из корня проекта:

```bash
cd /home/PavYtr/Projects/cv
```

Используется Conda-окружение `mmpose`. Системный `/usr/bin/python` для этого
проекта не подходит.

## 1. Проверка окружения и данных

Проверить MMPose и доступность CUDA:

```bash
conda run --no-capture-output -n mmpose python -c \
  "import torch, mmpose; print('MMPose:', mmpose.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Проверить, что подготовлены COCO-аннотации и изображения:

```bash
ls -lh data/LumbarCoco/annotations
test -d data/LumbarCoco/images && echo "Images: OK"
```

## 2. Запуск нового обучения с выводом в терминал

### Большой набор BUU-LSPINE_2000

Конвертация выполняется один раз. По умолчанию изображения не копируются:
создаётся ссылка на исходный каталог `LA`, поэтому дополнительные 3.9 ГБ не
занимаются.

```bash
conda run --no-capture-output -n mmpose python \
  tools/convert_buu_to_lumbar_coco.py \
  --source BUU-LSPINE_2000 \
  --output data/LumbarCoco2000
```

Получается фиксированный split с seed 42: 1600 train, 200 validation и 200
test. Для полной копии вместо ссылки добавьте `--image-mode copy`. Повторная
конвертация существующего каталога требует `--overwrite`.

Запуск обучения на 2000 снимках:

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco2000_1stage.py \
  --work-dir work_dirs/rtmpose-m-lumbar-coco2000-210e \
  --amp
```

### Малый набор BUU-LSPINE_400

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  --work-dir work_dirs/rtmpose-m_lumbar-coco-210e \
  --amp
```

`--no-capture-output` обязателен для живого вывода через `conda run`. Без него
Conda может удерживать весь stdout до завершения процесса. В терминале будут
видны эпоха и итерация, `loss_kpt`, `acc_pose`, learning rate, ETA и расход
GPU-памяти. Каждые 10 эпох выводятся AP и PCK и сохраняется checkpoint.

Вместо `conda run` можно сначала активировать окружение:

```bash
conda activate mmpose
python tools/train.py configs/rtmpose-m_lumbar-coco_1stage.py \
  --work-dir work_dirs/rtmpose-m_lumbar-coco-210e \
  --amp
```

Для нового запуска намеренно выбран новый каталог с суффиксом `-210e`:
существующий `work_dirs/rtmpose-m_lumbar-coco` уже содержит обучение на 420
эпох. Не запускайте новый эксперимент в старом каталоге — checkpoint с
совпадающими именами могут быть перезаписаны. Для BUU-2000 используется
`work_dirs/rtmpose-m-lumbar-coco2000-210e`, как в команде выше.

### Продолжение прерванного обучения

С тем же `--work-dir`:

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  --work-dir work_dirs/rtmpose-m_lumbar-coco-210e \
  --amp \
  --resume
```

`--resume` восстанавливает модель, оптимизатор, scheduler и номер эпохи из
checkpoint, указанного в файле `last_checkpoint`. Используйте только свои или
официальные доверенные checkpoint: MMEngine хранит в них не только веса, но и
Python-метаданные оптимизатора. Возобновлять следует с тем же конфигом и только
пока сохранённая эпоха меньше `MAX_EPOCHS`.

### Просмотр текстового лога из другого терминала

```bash
LOG=$(find work_dirs/rtmpose-m_lumbar-coco-210e -type f -name '*.log' \
  -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
tail -f "$LOG"
```

`Ctrl+C` здесь завершает только `tail`, но не обучение.

## 3. TensorBoard: графики обучения

Новые запуски записывают одновременно `scalars.json` и TensorBoard event-файл.
Запустить сервер:

```bash
conda run --no-capture-output -n mmpose tensorboard \
  --logdir /home/PavYtr/Projects/cv/work_dirs \
  --port 6006
```

Открыть в браузере:

```text
http://127.0.0.1:6006
```

В TensorBoard будут доступны loss, accuracy, learning rate, AP и четыре PCK:
`PCK@0.05`, `PCK@0.03`, `PCK@0.02`, `PCK@0.01`. Важно указывать каталог
`work_dirs` с буквой `s`, а не `work_dir`.

## 4. Выбор последней и лучшей модели

Сначала выберите каталог эксперимента. Для уже завершённого обучения на 420
эпохах:

```bash
EXPERIMENT_DIR=work_dirs/rtmpose-m_lumbar-coco
```

Для нового запуска из этой инструкции после его завершения используйте:

```bash
EXPERIMENT_DIR=work_dirs/rtmpose-m_lumbar-coco-210e
```

Последний периодический checkpoint выбранного эксперимента:

```bash
LAST_CHECKPOINT=$(cat "$EXPERIMENT_DIR/last_checkpoint")
echo "$LAST_CHECKPOINT"
```

Последний по времени checkpoint, признанный лучшим по PCK@0.05:

```bash
BEST_CHECKPOINT=$(find "$EXPERIMENT_DIR" -maxdepth 1 \
  -type f -name 'best_*.pth' -printf '%T@ %p\n' \
  | sort -nr | head -n1 | cut -d' ' -f2-)
echo "$BEST_CHECKPOINT"
```

Обычно для инференса следует использовать `$BEST_CHECKPOINT`, а не последний
checkpoint: последняя эпоха не обязана давать лучшую валидационную метрику.

## 5. Расчёт метрик последней или лучшей модели

Для последней модели:

```bash
conda run --no-capture-output -n mmpose python tools/test.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$LAST_CHECKPOINT" \
  --work-dir "$EXPERIMENT_DIR/test-last"
```

Для лучшей модели замените `"$LAST_CHECKPOINT"` на `"$BEST_CHECKPOINT"`.
Команда выводит COCO AP/AR и PCK на test-разбиении. Не передавайте в launcher
непроверенные `.pth` из сторонних источников.

## 6. Визуализация предсказаний модели

### Ground truth и prediction на одном изображении

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$BEST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --show-labels \
  --output output/comparisons/best_gt_vs_prediction.jpg
```

Зелёные окружности — ground truth, пурпурные кресты — prediction, светлые
отрезки показывают ошибку между ними.

### Ground truth и prediction рядом

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_and_prediction_side_by_side.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$BEST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --show-labels \
  --output output/comparisons/best_side_by_side.jpg
```

Если путь к изображению не передан, используется первое изображение из JSON.
Для конкретного снимка передайте его после checkpoint:

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$BEST_CHECKPOINT" \
  data/LumbarCoco/images/0149-M-062Y1.jpg \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --show-labels
```

Изображение должно входить в выбранный annotation JSON. Для снимков test
используйте `lumbar_keypoints_test.json`, для validation —
`lumbar_keypoints_val.json`.

### Сравнение последней и лучшей моделей

Запустите визуализатор дважды с одним и тем же изображением и разными файлами:

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction.py \
  configs/rtmpose-m_lumbar-coco_1stage.py "$LAST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --output output/comparisons/last_model.jpg

conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction.py \
  configs/rtmpose-m_lumbar-coco_1stage.py "$BEST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json \
  --output output/comparisons/best_model.jpg
```

Результаты сохраняются в `output/comparisons/`.

## 7. Где находятся результаты

```text
work_dirs/rtmpose-m_lumbar-coco/
├── last_checkpoint
├── epoch_*.pth
├── best_*.pth
└── YYYYMMDD_HHMMSS/
    ├── *.log
    └── vis_data/
        ├── scalars.json
        └── events.out.tfevents.*

output/comparisons/
└── *.jpg
```

Конфигурация сохраняет лучший checkpoint по `PCK@0.05/PCK`. Периодические
checkpoint создаются каждые 10 эпох, одновременно сохраняются не более трёх.
