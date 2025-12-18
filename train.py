import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HandSegmentationDataset
from unet import UNet
from tqdm import tqdm

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
dataset = HandSegmentationDataset(
    image_dir="data/images",
    mask_dir="data/masks",
    img_size=128
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# -----------------------------
# MODEL
# -----------------------------
model = UNet().to(device)

# -----------------------------
# LOSS FUNCTIONS
# -----------------------------
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAINING LOOP
# -----------------------------
epochs = 45

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    loop = tqdm(loader)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(loader):.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "hand_segmentation_unet.pth")
print("✅ Model saved as hand_segmentation_unet.pth")
