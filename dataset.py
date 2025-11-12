# dataset.py
import os
from glob import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
from config import IMAGES_PATH, TRAINING_PATH, VALIDATION_PATH, IMG_SIZE, TRAIN_SPLIT

# ------------------------------
# Class Mapping
# ------------------------------
dirs = sorted([d for d in glob(os.path.join(IMAGES_PATH, "*/"))])
num_classes = {}
for i, d in enumerate(dirs):
    class_name = os.path.basename(os.path.normpath(d))
    class_name = class_name.replace(" ", "_")
    num_classes[class_name] = i


def get_class(idx):
    inv = {v: k for k, v in num_classes.items()}
    return inv.get(idx, None)


def _safe_list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    items = []
    for p in glob(os.path.join(folder, "*")):
        if os.path.isfile(p) and p.lower().endswith(exts):
            items.append(os.path.basename(p))
    items.sort()
    return items


# ------------------------------
# Image Enhancement with OpenCV
# ------------------------------
def enhance_image_cv(img_path):
    """Apply mild OpenCV-based preprocessing before training."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Histogram equalization (improves contrast)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return Image.fromarray(img)


# ------------------------------
# Preprocessing: Split + Mean/Std
# ------------------------------
def preprocessing():
    """
    Generate training/validation CSVs and compute dataset mean/std.
    """
    train_lines, val_lines = [], []

    for class_name in sorted(num_classes.keys()):
        folder = os.path.join(IMAGES_PATH, class_name)
        if not os.path.isdir(folder):
            continue
        imgs = _safe_list_images(folder)
        random.shuffle(imgs)

        split = int(len(imgs) * TRAIN_SPLIT)
        train_imgs = imgs[:split]
        val_imgs = imgs[split:]

        train_lines += [f"{fname},{class_name}" for fname in train_imgs]
        val_lines += [f"{fname},{class_name}" for fname in val_imgs]

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    with open(TRAINING_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))

    # --- Compute mean/std ---
    print("Computing mean/std over training images (this may take time)...")
    t = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    sum_rgb = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0

    for line in train_lines:
        fname, cls = line.split(",")
        p = os.path.join(IMAGES_PATH, cls, fname)
        try:
            img = enhance_image_cv(p)
            if img is None:
                continue
            tensor = t(img)
            sum_rgb += tensor.sum((1, 2))
            sum_sq += (tensor ** 2).sum((1, 2))
            count += tensor.shape[1] * tensor.shape[2]
        except Exception as e:
            print(f"Warning, skipping {p}: {e}")

    mean = sum_rgb / count
    var = (sum_sq / count) - (mean ** 2)
    std = torch.sqrt(var)

    with open("mean_devstd.txt", "w") as f:
        f.write(",".join([str(x.item()) for x in mean]) + "," + ",".join([str(x.item()) for x in std]))

    print("✅ Dataset split complete & mean/std saved to mean_devstd.txt")


# ------------------------------
# Dataset Class
# ------------------------------
class LocalDataset(Dataset):
    def __init__(self, base_path, txt_list, transform=None):
        self.base_path = base_path
        if not os.path.exists(txt_list):
            raise FileNotFoundError(f"{txt_list} not found. Run preprocessing() first.")
        self.transform = transform

        data = []
        with open(txt_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                fname = parts[0]
                cls = ",".join(parts[1:])
                data.append((fname, cls))
        self.images = data

    def __getitem__(self, index):
        fname, cls = self.images[index]
        img_path = os.path.join(self.base_path, cls, fname)
        img = enhance_image_cv(img_path)
        if img is None:
            img = Image.new("RGB", IMG_SIZE, (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        label = num_classes[cls]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long), "img_name": fname}

    def __len__(self):
        return len(self.images)


# ------------------------------
# Train / Validation Transforms
# ------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
