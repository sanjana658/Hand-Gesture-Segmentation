# Hand Gesture Segmentation using U-Net

This project implements a **hand and arm region segmentation system** using a U-Net architecture. The model performs pixel-level segmentation to separate the hand (including wrist and forearm) from complex backgrounds such as outdoor scenes.

The project demonstrates a complete **computer vision segmentation pipeline**, from dataset preparation to training, inference, and result visualization.

---

Project Overview

- **Task**: Binary image segmentation
- **Model**: U-Net (encoder–decoder with skip connections)
- **Input**: RGB images
- **Output**: Binary segmentation mask (hand + arm region)
- **Framework**: PyTorch

---

##  Why U-Net?

U-Net is well-suited for segmentation tasks because it:
- Captures spatial context through downsampling
- Preserves fine details via skip connections
- Works effectively with limited datasets

---

##  Project Structure
hand-gesture-segmentation/
│
├── data/
│ ├── images/ # Input images
│ └── masks/ # Ground truth masks
│
├── dataset.py # Dataset loader
├── unet.py # U-Net architecture
├── train.py # Training pipeline
├── predict.py # Inference & visualization
├── requirements.txt # Dependencies
└── README.md

---

## Training Details

- **Image size**: 128 × 128  
- **Epochs**: 30  
- **Loss Function**: Binary Cross-Entropy + Dice Loss  
- **Optimizer**: Adam  
- **Batch Size**: 4  

Dice Loss improves segmentation continuity and boundary quality.




