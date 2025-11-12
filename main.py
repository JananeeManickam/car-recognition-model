# import os
# import argparse
# import numpy as np
# from matplotlib import pyplot as plt
# import torch
# from torch import nn
# from torch.optim import SGD
# from torch.utils.data import DataLoader
# from torchvision import models, transforms
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# import seaborn as sns
# from PIL import Image
# import sys
# import torch
# torch.set_grad_enabled(False)  

# from config import *
# from dataset import (
#     LocalDataset,
#     preprocessing,
#     num_classes as num_classes_dict,
#     get_class,
#     get_train_transform,
#     get_val_transform
# )

# # -----------------------------
# # CLI arguments
# # -----------------------------
# parser = argparse.ArgumentParser(description="Car model recognition trainer & predictor")
# parser.add_argument("--update", action="store_true", help="Run preprocessing + training pipeline")
# parser.add_argument("-i", "--input", type=str, help="Predict on a given image path")
# parser.add_argument("-m", "--model", type=str, choices=["resnet18", "lan", "both"], default="both",
#                     help="Which model to use for prediction: resnet18, lan, or both")
# args = parser.parse_args()

# # -----------------------------
# # Environment setup
# # -----------------------------
# device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
# print("Using device:", device)

# def get_class_list():
#     inv = {v: k for k, v in num_classes_dict.items()}
#     return [inv[i] for i in range(len(inv))]

# CLASS_LIST = get_class_list()
# NUM_CLASSES = len(CLASS_LIST)

# # -----------------------------
# # Model definitions
# # -----------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         padding = kernel_size // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         mx, _ = torch.max(x, dim=1, keepdim=True)
#         cat = torch.cat([avg, mx], dim=1)
#         attn = self.sigmoid(self.conv(cat))
#         return x * attn


# class LANetBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#         self.se = SEBlock(out_ch, reduction=8)
#         self.sa = SpatialAttention()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.se(x)
#         x = self.sa(x)
#         return x


# class LANet(nn.Module):
#     def __init__(self, num_classes=100, base_channels=32):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )
#         self.layer1 = LANetBlock(base_channels, base_channels)
#         self.layer2 = LANetBlock(base_channels, base_channels * 2, stride=2)
#         self.layer3 = LANetBlock(base_channels * 2, base_channels * 4, stride=2)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(base_channels * 4, num_classes)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# # -----------------------------
# # Build/load model
# # -----------------------------
# def build_model(model_key, num_classes):
#     if model_key == "resnet18":
#         # model = models.resnet18(weights="IMAGENET1K_V1")
#         model = models.resnet18(weights=None)  # no pretrained weights
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model, "resnet18"
#     elif model_key == "lan":
#         model = LANet(num_classes=num_classes)
#         return model, "lanet"
#     else:
#         raise ValueError("Invalid model key")


# def save_model(model, model_name):
#     path = os.path.join(RESULTS_PATH, model_name)
#     os.makedirs(path, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(path, f"{model_name}.pt"))


# def load_model(model_key, num_classes):
#     model, name = build_model(model_key, num_classes)
#     path = os.path.join(RESULTS_PATH, name, f"{name}.pt")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{path} not found. Train the model first.")
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model


# # -----------------------------
# # Training logic
# # -----------------------------
# def train_model(model_key, num_classes, train_loader, val_loader, lr=LEARNING_RATE, epochs=EPOCHS):
#     model, name = build_model(model_key, num_classes)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

#     for epoch in range(epochs):
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         for batch in train_loader:
#             imgs, labels = batch['image'].to(device), batch['label'].to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * imgs.size(0)
#             correct += (outputs.argmax(1) == labels).sum().item()
#             total += imgs.size(0)

#         train_loss = total_loss / total
#         train_acc = correct / total

#         # Validation
#         model.eval()
#         val_loss, val_correct, val_total = 0, 0, 0
#         preds_all, labels_all = [], []
#         with torch.no_grad():
#             for batch in val_loader:
#                 imgs, labels = batch['image'].to(device), batch['label'].to(device)
#                 outputs = model(imgs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * imgs.size(0)
#                 preds = outputs.argmax(1)
#                 val_correct += (preds == labels).sum().item()
#                 val_total += imgs.size(0)
#                 preds_all += preds.cpu().numpy().tolist()
#                 labels_all += labels.cpu().numpy().tolist()

#         val_loss /= max(1, val_total)
#         val_acc = val_correct / max(1, val_total)
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)

#         print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

#         if len(labels_all) > 0:
#             f1 = f1_score(labels_all, preds_all, average='weighted')
#             acc = accuracy_score(labels_all, preds_all)
#             print(f"📊 ERROR METRIC REPORT for {name} (Epoch {epoch+1}):")
#             print(f"➡️  Val Accuracy: {acc:.4f}")
#             print(f"➡️  Weighted F1-score: {f1:.4f}")
#             sys.stdout.flush()
#         else:
#             print(f"⚠️ No validation samples found — skipping error metrics.")
#             sys.stdout.flush()

#     # Final confusion matrix
#     if len(labels_all) > 0:
#         cm = confusion_matrix(labels_all, preds_all)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=False, cmap="Blues")
#         plt.title(f"Confusion Matrix — {name}")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.tight_layout()
#         cm_path = os.path.join(RESULTS_PATH, f"{name}_confusion_matrix.png")
#         os.makedirs(RESULTS_PATH, exist_ok=True)
#         plt.savefig(cm_path)
#         plt.close()
#         print(f"📈Confusion matrix saved at: {cm_path}")

#     save_model(model, name)
#     print(f" {name} training completed & saved.")
    
#     return model


# # -----------------------------
# # Prediction
# # -----------------------------
# def predict(model_key, img_path):
#     model = load_model(model_key, NUM_CLASSES)
#     im = Image.open(img_path).convert("RGB")
#     t = get_val_transform()
#     x = t(im).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = model(x)
#         probs = torch.nn.functional.softmax(out, dim=1)[0]
#     top_probs, top_idxs = torch.topk(probs, 3)
#     for i, (p, idx) in enumerate(zip(top_probs, top_idxs)):
#         print(f"{i+1}. {CLASS_LIST[idx]} — {p*100:.2f}%")


# # -----------------------------
# # Main entry
# # -----------------------------
# if __name__ == "__main__":
#     if args.update:
#         print("Running preprocessing + training...")
#         preprocessing()
#         print("Preprocessing done. Building datasets with augmentations...")

#         train_set = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=get_train_transform())
#         val_set = LocalDataset(IMAGES_PATH, VALIDATION_PATH, transform=get_val_transform())
#         train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

#         for m in ["resnet18", "lan"]:
#             train_model(m, NUM_CLASSES, train_loader, val_loader)

#     if args.input:
#         print(f"\nPredictions for {args.input}:")
#         if args.model == "both":
#             print("\n-- ResNet18 --")
#             predict("resnet18", args.input)
#             print("\n-- LANet --")
#             predict("lan", args.input)
#         else:
#             predict(args.model, args.input)

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from PIL import Image
import sys
import cv2
import matplotlib.pyplot as plt


from config import *
from dataset import (
    LocalDataset,
    preprocessing,
    num_classes as num_classes_dict,
    get_class,
    get_train_transform,
    get_val_transform
)

# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Car model recognition trainer & predictor")
parser.add_argument("--update", action="store_true", help="Run preprocessing + training pipeline")
parser.add_argument("-i", "--input", type=str, help="Predict on a given image path")
parser.add_argument("-m", "--model", type=str, choices=["resnet18", "lan", "both"], default="both",
                    help="Which model to use for prediction: resnet18, lan, or both")
args = parser.parse_args()

# -----------------------------
# Environment setup
# -----------------------------
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
print("Using device:", device)

def get_class_list():
    inv = {v: k for k, v in num_classes_dict.items()}
    return [inv[i] for i in range(len(inv))]

CLASS_LIST = get_class_list()
NUM_CLASSES = len(CLASS_LIST)

# -----------------------------
# Model definitions
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid() #activation method

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn


class LANetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch, reduction=8)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.sa(x)
        return x


class LANet(nn.Module):
    def __init__(self, num_classes=100, base_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = LANetBlock(base_channels, base_channels)
        self.layer2 = LANetBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = LANetBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# Build/load model
# -----------------------------
def build_model(model_key, num_classes):
    if model_key == "resnet18":
        with torch.no_grad():
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "resnet18"
    elif model_key == "lan":
        with torch.no_grad():
            model = LANet(num_classes=num_classes)
        return model, "lanet"
    else:
        raise ValueError("Invalid model key")


def save_model(model, model_name):
    path = os.path.join(RESULTS_PATH, model_name)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f"{model_name}.pt"))


def load_model(model_key, num_classes):
    with torch.no_grad():
        model, name = build_model(model_key, num_classes)
        path = os.path.join(RESULTS_PATH, name, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Train the model first.")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    return model


# -----------------------------
# Training logic
# -----------------------------
def train_model(model_key, num_classes, train_loader, val_loader, lr=LEARNING_RATE, epochs=EPOCHS):
    model, name = build_model(model_key, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            imgs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                preds_all += preds.cpu().numpy().tolist()
                labels_all += labels.cpu().numpy().tolist()

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

        if len(labels_all) > 0:
            f1 = f1_score(labels_all, preds_all, average='weighted')
            acc = accuracy_score(labels_all, preds_all)
            print(f"📊 ERROR METRIC REPORT for {name} (Epoch {epoch+1}):")
            print(f"➡️  Val Accuracy: {acc:.4f}")
            print(f"➡️  Weighted F1-score: {f1:.4f}")
            sys.stdout.flush()
        else:
            print(f"⚠️ No validation samples found — skipping error metrics.")
            sys.stdout.flush()

    # Final confusion matrix
    if len(labels_all) > 0:
        cm = confusion_matrix(labels_all, preds_all)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_PATH, f"{name}_confusion_matrix.png")
        os.makedirs(RESULTS_PATH, exist_ok=True)
        plt.savefig(cm_path)
        plt.close()
        print(f"📈Confusion matrix saved at: {cm_path}")

    save_model(model, name)
    print(f"✅ {name} training completed & saved.")
    
    return model


# -----------------------------
# Prediction
# -----------------------------

def predict(model_key, img_path):

    # Step 1️: Basic reading + preprocessing
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Error: Image not found or cannot be opened.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # print("Step 1: Image loaded and converted to RGB")

    # Step 2️: Image resizing (consistent with transforms)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    # print(f"Step 2: Resized to {IMG_SIZE}")

    # Step 3️: Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    # print("Step 3: Converted to grayscale")

    # Step 4️: Segmentation using thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print("Step 4: Segmentation (Otsu thresholding) applied")

    # Step 5: Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # print("Step 5: Morphological closing applied")

    # Step 6: Edge detection (for visualization)
    edges = cv2.Canny(gray, 100, 200)
    # print("Step 6: Edge detection completed")

    # Save intermediate images
    os.makedirs(RESULTS_PATH, exist_ok=True)
    cv2.imwrite(os.path.join(RESULTS_PATH, "1_gray.png"), gray)
    cv2.imwrite(os.path.join(RESULTS_PATH, "2_thresh.png"), thresh)
    cv2.imwrite(os.path.join(RESULTS_PATH, "3_morph.png"), morph)
    cv2.imwrite(os.path.join(RESULTS_PATH, "4_edges.png"), edges)
    # print("---Intermediate images saved under /results")

    # Step 7️⃣: Classification (model prediction)
    print(f"\nRunning classification using {model_key.upper()} ...")
    model = load_model(model_key, NUM_CLASSES)
    im = Image.fromarray(img_rgb).convert("RGB")
    t = get_val_transform()
    x = t(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, 3)
    print("\nTop Predictions:")
    for i, (p, idx) in enumerate(zip(top_probs, top_idxs)):
        print(f"{i+1}. {CLASS_LIST[idx]} — {p*100:.2f}%")

    # Step 8️⃣: Display results (optional visualization)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].imshow(gray, cmap="gray"); axs[0].set_title("Gray")
    axs[1].imshow(thresh, cmap="gray"); axs[1].set_title("Threshold")
    axs[2].imshow(morph, cmap="gray"); axs[2].set_title("Morphology")
    axs[3].imshow(edges, cmap="gray"); axs[3].set_title("Edges")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "cv_pipeline.png"))
    plt.close()

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    if args.update:
        print("Running preprocessing + training...")
        preprocessing()
        print("Preprocessing done. Building datasets with augmentations...")

        train_set = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=get_train_transform())
        val_set = LocalDataset(IMAGES_PATH, VALIDATION_PATH, transform=get_val_transform())
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        for m in ["resnet18", "lan"]:
            train_model(m, NUM_CLASSES, train_loader, val_loader)

    if args.input:
        print(f"\nPredictions for {args.input}:")
        if args.model == "both":
            print("\n-- ResNet18 --")
            predict("resnet18", args.input)
            print("\n-- LANet --")
            predict("lan", args.input)
        else:
            predict(args.model, args.input)