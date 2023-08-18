import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from dataset_manager import PennFudanDataset

chemin = "C:\\Users\\Audensiel\\Desktop\\image_recognition_and_segmentation\\PennFudanPed"

dataset = PennFudanDataset(root=chemin, transforms=None)

print("Nombre total d'images ", len(dataset))

index = 0
image, target = dataset[index]

print("Image:", image.size)
print("Target:", target)
