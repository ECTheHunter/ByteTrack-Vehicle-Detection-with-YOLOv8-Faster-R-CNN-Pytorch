import os
import cv2
import albumentations as A
import shutil

# ⚙️ Set up your paths
IMAGE_DIR = 'UA-DETRAC/train/images'
LABEL_DIR = 'UA-DETRAC/train/labels'
OUTPUT_IMAGE_DIR = 'UA-DETRAC/train_aug/images'
OUTPUT_LABEL_DIR = 'UA-DETRAC/train_aug/labels'
for file in os.listdir(IMAGE_DIR):
    if '_aug' in file:
        os.remove(os.path.join(IMAGE_DIR, file))

# Delete augmented labels
for file in os.listdir(IMAGE_DIR):
    if '_aug' in file:
        os.remove(os.path.join(IMAGE_DIR, file))
# Create output dirs
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 📌 Classes to augment (YOLO class IDs)
TARGET_CLASSES = [0, 2, 3]  # Example: 5 = bus, 6 = truck, 7 = van (edit based on your dataset)

# 🧰 Albumentations augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.MotionBlur(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_label(label_path):
    boxes, class_ids = [], []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x, y, w, h = map(float, line.strip().split())
            if int(class_id) in TARGET_CLASSES:
                boxes.append([x, y, w, h])
                class_ids.append(int(class_id))
    return boxes, class_ids

def save_augmented(image, bboxes, class_ids, base_name, count):
    img_out = f"{OUTPUT_IMAGE_DIR}/{base_name}_aug{count}.jpg"
    lbl_out = f"{OUTPUT_LABEL_DIR}/{base_name}_aug{count}.txt"
    
    cv2.imwrite(img_out, image)
    with open(lbl_out, 'w') as f:
        for cls, box in zip(class_ids, bboxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")

# 🔄 Loop through dataset
for img_file in os.listdir(IMAGE_DIR):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    
    if not os.path.exists(label_path):
        continue

    # Read image and labels
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    boxes, class_ids = read_yolo_label(label_path)

    # Skip if no target classes
    if not boxes:
        continue

    # 🔁 Apply augmentation N times
    for i in range(2):  # how many times to augment each image
        try:
            augmented = transform(image=image, bboxes=boxes, class_labels=class_ids)
            aug_img = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_class_ids = augmented['class_labels']
            save_augmented(aug_img, aug_boxes, aug_class_ids, base_name, i)
        except Exception as e:
            print(f"Skipping {base_name} due to error: {e}")

def merge_folders(src, dst):
    for file in os.listdir(src):
        shutil.move(os.path.join(src, file), os.path.join(dst, file))

# Merge images and labels
merge_folders('UA-DETRAC/train_aug/images', 'UA-DETRAC/train/images')
merge_folders('UA-DETRAC/train_aug/labels', 'UA-DETRAC/train/labels')

# Optionally remove the now-empty 'train_aug' dirs
os.rmdir('UA-DETRAC/train_aug/labels')
os.rmdir('UA-DETRAC/train_aug/images')
