import urllib.request
import cv2
import torch
from mmpose.apis import MMPoseInferencer

# Проверяем GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Запуск на устройстве: {device} ({torch.cuda.get_device_name(0)})")

# 1. Скачиваем тестовую картинку локально
img_path = 'demo.jpg'
url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg'
print(">>> Скачиваем тестовую картинку...")
urllib.request.urlretrieve(url, img_path)

# 2. Инициализируем модель позы MMPose
print(">>> Загрузка нейросети MMPose (RTMPose)...")
inferencer = MMPoseInferencer('human', device=device)

# 3. Передаем рамку человека напрямую в MMPose (минуя сторонний mmdet)
img = cv2.imread(img_path)
h, w = img.shape[:2]
person_box = [20, 20, w - 20, h - 20]  # рамка вокруг объекта

print(">>> Нейросеть MMPose ищет ключевые точки скелета на RTX 5070 Ti...")
result_generator = inferencer(img_path, bboxes=[person_box], out_dir='output')
results = list(result_generator)

print("\n" + "="*60)
print(">>> ПОЛНАЯ ПОБЕДА! MMPOSE РАБОТАЕТ! <<<")
keypoints = results[0]['predictions'][0][0]['keypoints']
print(f"Успешно найдено ключевых точек: {len(keypoints)}")
print("Размеченная картинка сохранена в: output/visualizations/demo.jpg")
print("="*60)