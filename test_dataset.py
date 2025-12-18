from dataset import HandSegmentationDataset

dataset = HandSegmentationDataset(
    image_dir="data/images",
    mask_dir="data/masks"
)

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
