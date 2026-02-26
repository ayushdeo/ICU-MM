# This is a sample script to shoow how to work with the MIMIC CXR data


import pydicom
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# -----------------------------
# 1️ Load DICOM
# -----------------------------
file_path = r"p10000980\s57861150\5aa15ba6-55f5e96e-39cea686-7c3b28b2-b8c97a88.dcm"

ds = pydicom.dcmread(file_path)
img = ds.pixel_array.astype(np.float32)

print("Patient ID:", ds.PatientID)
print("Study Date:", ds.StudyDate)
print("Modality:", ds.Modality)
print("View Position:", ds.get("ViewPosition", "N/A"))
print("Photometric Interpretation:", ds.PhotometricInterpretation)

plt.figure(figsize=(6,8))
plt.imshow(img, cmap="gray")
plt.title("Raw DICOM")
plt.axis("off")
plt.show()

#MIMIC-IV (cohort.csv)
#    ↓ subject_id
#MIMIC-CXR patient folder
#    ↓ study_id
#Imaging study
#    ↓ DICOM + .txt
#Image + Report
# -----------------------------
# 2️ Apply DICOM scaling
# -----------------------------
slope = ds.get("RescaleSlope", 1)
intercept = ds.get("RescaleIntercept", 0)
img = img * slope + intercept


# -----------------------------
# 3️Fix MONOCHROME1 inversion
# -----------------------------
if ds.PhotometricInterpretation == "MONOCHROME1":
    img = np.max(img) - img

# ----------------------------
# 4️ Percentile normalization (for CXR)
# -----------------------------
lower = np.percentile(img, 1)
upper = np.percentile(img, 99)
img_norm = np.clip(img, lower, upper)
img_norm = (img_norm - lower) / (upper - lower)

plt.figure(figsize=(6,8))
plt.imshow(img_norm, cmap="gray")
plt.title("Normalized CXR")
plt.axis("off")
plt.show()
print("Intensity range before normalization:", img.min(), img.max())

# -----------------------------
# 5️ Prepare model tensor
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor_img = transform((img_norm * 255).astype(np.uint8))

print("Tensor shape:", tensor_img.shape)
print("Tensor min/max:", tensor_img.min().item(), tensor_img.max().item())

# --------------------------------------------
# redo of the code by gpt to generate images side by side
# --------------------------------------------
import pydicom
import numpy as np
import matplotlib.pyplot as plt

file_path = r"p10000980\s57861150\5aa15ba6-55f5e96e-39cea686-7c3b28b2-b8c97a88.dcm"

ds = pydicom.dcmread(file_path)
raw = ds.pixel_array.astype(np.float32)

# 1️ DICOM scaling
slope = ds.get("RescaleSlope", 1)
intercept = ds.get("RescaleIntercept", 0)
scaled = raw * slope + intercept

# 2️ Fix inversion if needed
if ds.PhotometricInterpretation == "MONOCHROME1":
    inverted = np.max(scaled) - scaled
else:
    inverted = scaled.copy()

# 3️ Percentile normalization
lower = np.percentile(inverted, 1)
upper = np.percentile(inverted, 99)
normalized = np.clip(inverted, lower, upper)
normalized = (normalized - lower) / (upper - lower)

# -----------------------------
# Plot side by side
# -----------------------------
fig, axs = plt.subplots(1, 4, figsize=(18,6))

axs[0].imshow(raw, cmap="gray")
axs[0].set_title("Raw DICOM")
axs[0].axis("off")

axs[1].imshow(scaled, cmap="gray")
axs[1].set_title("After Scaling")
axs[1].axis("off")

axs[2].imshow(inverted, cmap="gray")
axs[2].set_title("After Inversion Fix")
axs[2].axis("off")

axs[3].imshow(normalized, cmap="gray")
axs[3].set_title("Final Normalized")
axs[3].axis("off")

plt.tight_layout()
plt.show()