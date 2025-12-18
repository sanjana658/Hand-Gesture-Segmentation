import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 160
MODEL_PATH = "hand_segmentation_unet.pth"
IMAGE_PATH = "data/images/image (39).jpg"  # change image name to test others

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# READ IMAGE
# -----------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# PREPROCESS
# -----------------------------
image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
image_norm = image_resized / 255.0

input_tensor = torch.tensor(image_norm, dtype=torch.float32) \
    .permute(2, 0, 1) \
    .unsqueeze(0) \
    .to(device)

# -----------------------------
# PREDICT
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)
    mask = torch.sigmoid(output).cpu().numpy()[0, 0]

# -----------------------------
# THRESHOLD (IMPORTANT)
# -----------------------------
binary_mask = (mask > 0.3).astype(np.uint8)

# -----------------------------
# MORPHOLOGICAL SMOOTHING
# -----------------------------
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# -----------------------------
# RESIZE MASK TO ORIGINAL IMAGE
# -----------------------------
mask_display = cv2.resize(
    binary_mask,
    (image.shape[1], image.shape[0])
)

# -----------------------------
# OVERLAY MASK
# -----------------------------
overlay = image_rgb.copy()
overlay[mask_display == 1] = [255, 0, 0]  # red hand

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(mask_display, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# OPTIONAL: SAVE OUTPUTS
# -----------------------------
cv2.imwrite("output_mask.png", mask_display * 255)
cv2.imwrite("output_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
