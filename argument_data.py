import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

# Caminhos
input_images = glob("C:\\Users\\Johna\\Downloads\\mask_weared_incorrect.yolov8\\train\\argumentattion\\*.jpg")
input_labels = glob("C:\\Users\\Johna\\Downloads\\mask_weared_incorrect.yolov8\\train\\argumentattion\\*.txt")
output_images = "C:\\Users\\Johna\\Downloads\\mask_weared_incorrect.yolov8\\train\\argumentattion\\images2"
output_labels = "C:\\Users\\Johna\\Downloads\\mask_weared_incorrect.yolov8\\train\\argumentattion\\labels2"

# Cria pastas de saída se não existirem
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(shear=10, p=0.3),
    A.Blur(blur_limit=3, p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Processa cada imagem
for img_path, label_path in zip(input_images, input_labels):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Linha inválida em {label_path}: {line.strip()}")
            continue
        try:
            class_id, x_center, y_center, width, height = map(float, parts)
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
        except ValueError as e:
            print(f"Erro ao processar linha em {label_path}: {line.strip()}")
            print(f"Erro: {e}")
            continue

    # Aplica augmentation 3x por imagem
    for i in range(3):
        transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        if len(transformed["bboxes"]) > 0:
            new_img_path = os.path.join(output_images, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i}.jpg")
            cv2.imwrite(new_img_path, cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR))

            new_label_path = os.path.join(output_labels, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i}.txt")
            with open(new_label_path, "w") as f:
                for bbox, cls in zip(transformed["bboxes"], transformed["class_labels"]):
                    x_center, y_center, width, height = bbox
                    f.write(f"{cls} {x_center} {y_center} {width} {height}\n")