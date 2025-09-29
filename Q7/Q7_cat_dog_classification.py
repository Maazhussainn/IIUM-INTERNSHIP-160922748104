# Q7. Cat vs Dog Classification with ResNet50 (ImageNet Pretrained)

import os
import shutil
import csv
import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms

# -------------------------
# SETTINGS
# -------------------------
IMAGE_FOLDER = "/content/drive/MyDrive/Assignment internship/data/cat_dog/images"  # uploaded folder
OUT_CSV = "/content/drive/MyDrive/Assignment internship/result/Q7/Q7_report.csv"
MIS_TXT = "/content/drive/MyDrive/Assignment internship/result/Q7/Q7_misleading.txt"
MIS_IMG_DIR = "/content/drive/MyDrive/Assignment internship/result/Q7/misleading_images"

# -------------------------
# Load model and categories
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights).to(device).eval()
categories = weights.meta["categories"]

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------
# Prediction function
# -------------------------
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out[0], dim=0)
        top1_prob, top1_idx = torch.max(prob, dim=0)
    return categories[int(top1_idx.item())], float(top1_prob), int(top1_idx.item())

def is_dog_index(idx: int) -> bool:
    '''ImageNet dog classes are indices 151..268 (inclusive).'''
    return 151 <= idx <= 268

# -------------------------
# Run classification
# -------------------------
results = []
misleading = []

files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(files)} images\n")

for fname in files:
    path = os.path.join(IMAGE_FOLDER, fname)
    label, prob, idx = predict(path)
    print(f"{fname} | Pred={label} ({prob*100:.1f}%)")
    results.append([fname, label, f"{prob:.4f}"])
    if not is_dog_index(idx):  # if prediction is not a dog
        misleading.append((fname, label, prob))

# -------------------------
# Save CSV results
# -------------------------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "prediction", "probability"])
    writer.writerows(results)

print(f"\n Report saved at: {OUT_CSV}")

# Save misleading list
if misleading:
    with open(MIS_TXT, "w", encoding="utf-8") as f:
        f.write("Misleading images (predicted as non-dog):\n")
        for fname, label, prob in misleading:
            f.write(f"{fname} -> {label} ({prob*100:.2f}%)\n")
    print(f" Misleading list saved at: {MIS_TXT}")

    os.makedirs(MIS_IMG_DIR, exist_ok=True)
    for fname, _, _ in misleading[:5]:  # save first 5 misleading examples
        src = os.path.join(IMAGE_FOLDER, fname)
        dst = os.path.join(MIS_IMG_DIR, fname)
        shutil.copy2(src, dst)
    print(f" Mislabel images copied to: {MIS_IMG_DIR}")
else:
    print(" No misleading images found (all predicted as dogs).")

# -------------------------
# Summary
# -------------------------
total = len(files)
dog_preds = total - len(misleading)
accuracy = (dog_preds / total) * 100 if total > 0 else 0

print("\n--- SUMMARY ---")
print(f"Total images: {total}")
print(f"Predicted as dogs: {dog_preds}")
print(f"Misleading (non-dogs): {len(misleading)}")
print(f"Dog classification accuracy: {accuracy:.2f}%")
