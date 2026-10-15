# Запуск обучения и визуализаций RTMPose-m / LumbarCoco


## Парный эксперимент: полный снимок и эмпирический RandomVertebraeCrop

Новые конфиги:

- `configs/rtmpose-m_lumbar-coco2000_fullimage.py` — без случайных геометрических аугментаций.
- `configs/rtmpose-m_lumbar-coco2000_vertebrae-crop.py` — с `RandomVertebraeCrop` и JSON статистики.

Они наследуют одинаковую модель, 210 эпох, batch size 64, seed 21,
оптимизатор, scheduler, EMA и Local PCK из существующего эксперимента.
Различается только train pipeline. Старые конфиги сохранены для воспроизводимости.

`FullImageBBox` после загрузки изображения заменяет аннотационный bbox на
`[0, 0, width, height]` (xyxy) во всех новых pipeline. Исходные COCO JSON
не переписываются. Padding равен 1.0. Валидация, test и pipeline,
который MMPose использует для инференса, одинаковы: полный снимок,
детерминированное приведение к размеру входа, без вертикализации по GT.
`TopdownAffine` расширяет прямоугольник до соотношения сторон 256:512,
поэтому весь снимок сохраняется, а дополнительные поля заполняются чёрным.
Для инференса через `inference_topdown` передавайте один bbox полного снимка,
чтобы получить один набор предсказаний без детектора.

### Статистика только обучающей части

```bash
conda run --no-capture-output -n mmpose python tools/collect_vertebrae_statistics.py \
  --annotations data/LumbarCoco2000/annotations/lumbar_keypoints_train.json \
  --output reports/vertebrae_statistics_train.json
```

Порядок 22 точек проверяется по `categories.keypoints` COCO; он совпадает
с конвертером и `configs/_base_/datasets/lumbar_coco.py`.
Середина верхней пластины L1: `L1TF/L1TB`, индексы 0/1;
середина крестца: `S1TF/S1TB`, индексы 20/21 (индексы с нуля).
Угол — `atan2(S1.x-L1.x, S1.y-L1.y)` в градусах: положительный,
если середина S1 расположена правее середины L1.

Вертикализация — жёсткий поворот всех точек и прямоугольника исходного
снимка. Четыре отступа измеряются как минимальные расстояния вдоль
соответствующей оси от стороны bbox всех размеченных точек до границы
реального повёрнутого снимка. Используется его четырёхугольник, а не
прямоугольная рамка расширенного canvas: добавленные чёрные треугольники
не считаются отступами. Имеющиеся чёрные участки самого снимка считаются
его частью; интенсивность пикселей не используется для определения границ.

JSON содержит угол, отступы в пикселях, размеры области точек и относительные
отступы (слева/справа делятся на её ширину, сверху/снизу — на высоту).
Углы bbox области точек иногда выходят за повёрнутый четырёхугольник,
хотя сами точки внутри: отрицательный доступный отступ записывается
в `signed_clearance_px`, а для вычисления среднего отступа используется ноль.
Также сохраняются идентификаторы снимков, SHA256 аннотаций и причины
пропущенных записей. Собранная версия: 1600 записей, 0 пропусков.
Раздел `mean` содержит арифметическое среднее отступов каждой стороны
в пикселях и относительных единицах, средний знаковый угол `angle_deg`
и средний абсолютный угол `absolute_angle_deg`. Относительные отступы
усредняются напрямую, а не делением средних пикселей на средний размер.
Средние относительные отступы: left=1.942013, right=1.969210,
top=0.348893, bottom=0.662390. Средний угол −2.526957°,
средний абсолютный угол 5.954573°.

### Геометрия аугментации

Вся реализация `RandomVertebraeCrop`, включая геометрические вычисления,
находится в одном модуле `augmentations/vertebrae_crop.py`. Скрипт статистики
использует функции из того же модуля.

`RandomVertebraeCrop` читает средние из JSON и формирует параметры:
`expansion_ranges[side] = (0, mean.margins_relative[side])`,
`rotation_range = (-mean.absolute_angle_deg, +mean.absolute_angle_deg)`.
Каждый отступ и угол выбираются независимо и равномерно из этих диапазонов.
Можно передать `expansion_ranges` и `rotation_range` напрямую вместо JSON
или переопределить один из них при загрузке JSON. Например:

```python
dict(type='RandomVertebraeCrop',
     expansion_ranges=dict(left=(0, 1.942013), right=(0, 1.969210),
                           top=(0, 0.348893), bottom=(0, 0.662390)),
     rotation_range=(-5.954573, 5.954573))
```

Исходный снимок виртуально вертикализируется, после чего
вводится выбранный наклон. Кроп охватывает все размеченные точки при новом
наклоне; отступы масштабируются по ширине/высоте вертикализованной области
точек текущего снимка. Дополнительно сохраняется запас 2 исходных пикселя
с каждой стороны. Устанавливаются только `bbox_center`, `bbox_scale` и
`bbox_rotation`; исходные пиксели и координаты точек не меняются на этом шаге.
Единственный `TopdownAffine` одновременно выполняет поворот, кроп,
масштабирование и преобразование точек с чёрным заполнением.
`RandomBBoxTransform` и `RandomLumbarBBoxExpansion` в новых pipeline отсутствуют.
Коррекция aspect ratio может увеличить полученный кроп, поэтому итоговые
отступы на входе модели могут быть больше выбранных из статистики.
Изгиб позвоночника сохраняется: итоговая геометрия — перенос, поворот и
равномерное масштабирование.

### Проверки и визуализация

```bash
conda run --no-capture-output -n mmpose python -m unittest discover -s tests -v
conda run --no-capture-output -n mmpose python tools/visualize_augmentation.py \
  configs/rtmpose-m_lumbar-coco2000_vertebrae-crop.py \
  --index 100 --seed 142 \
  --output output/visualizations/vertebrae_crop/sample_100.jpg
```

Проверяется сохранение всех точек для 1600 обучающих записей,
выбранный угол, отсутствие деформации, реальные границы вместо canvas,
равенство настроек экспериментов, независимость валидации от GT,
а также полный pipeline с генерацией SimCC targets на пяти снимках.
До/после сохранены для индексов 0, 7, 100, 800, 1599 в
`output/visualizations/vertebrae_crop/` (seed = 42 + index).
Слева — весь исходный снимок с детерминированным приведением к размеру
входа; справа — результат нового train pipeline. На обеих панелях точки.

### Обучение (команды подготовлены; обучение не запускалось)

Запускать из корня репозитория, после сбора статистики:

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco2000_fullimage.py \
  --work-dir work_dirs/rtmpose-m-lumbar-coco2000-fullimage --amp

conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco2000_vertebrae-crop.py \
  --work-dir work_dirs/rtmpose-m-lumbar-coco2000-vertebrae-crop --amp
```


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

### BUU-LSPINE_2000: обучение с bbox-аугментацией и Local PCK

Текущий конфиг расширяет выбранные стороны bbox на случайную величину от 0 до
50% его исходной ширины или высоты, обрезая расширение по краям изображения.
При применении аугментации crop поворачивается на случайный угол от −15° до
+15°. Валидация и test считают только `LocalPCK/segment@0.1/PCK`; по этой же
метрике сохраняется лучший checkpoint. COCO AP и PCK по bbox в этом конфиге
не рассчитываются.

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py \
  --work-dir work_dirs/rtmpose-m-lumbar-coco2000-bbox-localpck \
  --amp
```

Используйте тот же конфиг и каталог при возобновлении обучения, добавив
`--resume`. Для этого конфига валидация и сохранение checkpoint выполняются
каждые 10 эпох из 210.

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

### Сравнение изображения до и после аугментации

Команда сохраняет два входных crop рядом: слева стандартное кадрирование без
случайной аугментации, справа результат обучающего конвейера. На обоих кадрах
показаны ключевые точки и связи между ними. Checkpoint и GPU не нужны.

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_augmentation.py \
  configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py \
  --index 0 --seed 42 \
  --output output/visualizations/augmentation_comparison.jpg
```

`--index` выбирает пример из обучающего набора, `--seed` позволяет повторить
тот же случайный результат. Для обычной конфигурации можно указать
`configs/rtmpose-m_lumbar-coco_1stage.py`. Если точки мешают просмотру,
добавьте `--hide-keypoints`.

Чтобы получить 20 сравнений во время обучения, запустите:

```bash
conda run --no-capture-output -n mmpose python tools/train.py \
  configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py \
  --work-dir work_dirs/rtmpose-m-lumbar-bbox-vis \
  --amp --augmentation-samples 20
```

Снимки появятся в `work_dirs/rtmpose-m-lumbar-bbox-vis/augmentation_comparisons/`
через равные интервалы обучения. Каждый снимок использует отдельный пример из
train-набора и отдельную случайную аугментацию; это просмотр обучающего
конвейера, а не точная копия текущего minibatch. При `--resume` уже сохранённые
снимки остаются в каталоге, новые создаются на оставшихся контрольных шагах.

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
При обучении с bbox-аугментацией вместо AP и этих PCK будет записываться
`LocalPCK/segment@0.1/PCK`.

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

### Test после обучения с bbox-аугментацией

Из корня проекта найдите лучший checkpoint именно в каталоге нового запуска и
запустите test на 200 снимках из `data/LumbarCoco2000`:

```bash
EXPERIMENT_DIR=work_dirs/rtmpose-m-lumbar-coco2000-bbox-localpck
BEST_CHECKPOINT=$(find "$EXPERIMENT_DIR" -maxdepth 1 \
  -type f -name 'best_*.pth' -printf '%T@ %p\n' \
  | sort -nr | head -n1 | cut -d' ' -f2-)
if test -n "$BEST_CHECKPOINT" && test -f "$BEST_CHECKPOINT"; then
  conda run --no-capture-output -n mmpose python tools/test.py \
    configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py \
    "$BEST_CHECKPOINT" \
    --work-dir "$EXPERIMENT_DIR/test-best"
else
  echo "Лучший checkpoint не найден в $EXPERIMENT_DIR"
fi
```

Результат test — `LocalPCK/segment@0.1/PCK`. Для последнего checkpoint вместо
поиска лучшего используйте `LAST_CHECKPOINT=$(cat "$EXPERIMENT_DIR/last_checkpoint")`
и передайте `"$LAST_CHECKPOINT"` в ту же команду.

## 6. Визуализация предсказаний модели

### Визуализация случайных снимков test

```bash
conda run --no-capture-output -n mmpose python tools/visualize_test_grid.py
```

Команда использует лучший checkpoint из
`work_dirs/rtmpose-m-lumbar-coco2000-randombbox-localpck`, сохраняет 10
отдельных изображений и общий `grid.jpg` с увеличенными областями позвоночника
в `output/test_visualizations/<checkpoint>/seed_<число>_count_10/`.
Оранжевые окружности показывают радиус допуска LocalPCK, пурпурные кресты и
линии — предсказания. При каждом запуске
выбираются новые снимки; использованный seed печатается в терминале.
`--seed 42` воспроизведёт подборку, `--count 20` изменит число снимков,
`--columns 3` — число столбцов в общем изображении, а `--output-dir` задаст
папку сохранения.

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
Для визуализации с LocalPCK используйте отдельные команды:

```bash
conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_vs_prediction_local_pck.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$BEST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json

conda run --no-capture-output -n mmpose python \
  tools/visualize_gt_and_prediction_side_by_side_local_pck.py \
  configs/rtmpose-m_lumbar-coco_1stage.py \
  "$BEST_CHECKPOINT" \
  --annotations data/LumbarCoco/annotations/lumbar_keypoints_val.json
```

По умолчанию выводится `LocalPCK/segment@0.05`. Для другого режима или порога
передайте `--local-pck-mode image_mean` и `--local-pck-thr 0.1`. Метрика
считается по видимым GT-точкам в валидных сегментах, как при оценке модели;
`--score-thr` влияет только на отображение предсказанных точек.
Оранжевые окружности в обоих вариантах показывают допустимый радиус ошибки
вокруг GT-точки: порог, умноженный на длину её GT-сегмента (или на среднюю
длину валидных сегментов при `image_mean`). В этих вариантах GT-точки и зелёный
скелет не рисуются: положение GT задают центры оранжевых окружностей.

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
Для `configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py` рассчитывается только
`LocalPCK/segment@0.1/PCK`; лучший checkpoint выбирается по этой метрике.
