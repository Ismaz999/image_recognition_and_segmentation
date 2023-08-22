import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from dataset_manager import PennFudanDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from torch.optim.lr_scheduler import StepLR
from torch import inf
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


# Chemin vers le répertoire contenant les données (images, masques)
chemin = "C:\\Users\\Audensiel\\Desktop\\image_recognition_and_segmentation\\PennFudanPed"

###################################
#print("Nombre total d'images ", len(dataset))
#index = 0
#image, target = dataset[index]
#print("Image:", image.size)
#print("Target:", target)
####################################


def chargement_model(num_classes):
    # On vient charger le modèle maskrcnn resnet50 préentraîné sur le dataset COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # On récupère le nombre d'inputs du modèle resnet pour la prédiction des bounding box
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Puis on définit le nombre d'entrées du modèle qu'on va venir fine-tuner pour que ça corresponde à notre besoin
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # On vient modifier la tête de prédiction des masques de segmentation pour le nombre de classes souhaité
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # On définit le nombre d'entrées du nouveau modèle pour la prédiction des masques
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model



def get_transform(train):
    transforms = []
    # Convertit l'image d'entrée en un tenseur PyTorch (nécessaire pour traiter les images avec PyTorch)
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5)) # Crée des images miroirs pour augmenter le nombre de données
    return T.Compose(transforms)

################TEST DES FONCTIONS###################
#num_classes = 10
#model = chargement_model(num_classes)  
#print(model)
#model.eval()
#input_image = torch.randn(1, 3, 256, 256) 
#predictions = model(input_image)
#print(predictions)
#####################################################

# Créer des transformations pour les ensembles d'entraînement et de test
transform_train = get_transform(train=True) 
transform_test = get_transform(train=False)

dataset = PennFudanDataset(root=chemin, transforms=transform_train)
dataset_test = PennFudanDataset(root=chemin, transforms=transform_test)

# On divise les ensembles de données en ensembles d'entraînement et de test

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# On crée un DataLoader pour l'ensemble de données d'entraînement 
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0, #batch_size = le nombre de de données par lot, num_workers = le nombre de processus en parallèle pour charger les données
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


# device = torch.device('cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"

####################

# print("Total number of samples in dataset:", len(dataset))
# sample_image, sample_target = dataset[0]  # Get the first sample
# print("Sample image size:", sample_image.size)
# print("Sample target:", sample_target)

# print("Train dataset length:", len(dataset))
# print("Test dataset length:", len(dataset_test))

# for batch in data_loader:
#     images, targets = batch
#     print("Train batch images:", images)
#     print("Train batch targets:", targets)

# for batch in data_loader_test:
#     images, targets = batch
#     print("Test batch images:", images)
#     print("Test batch targets:", targets)

##################

device = torch.device('cpu')

 #Chargement du modèle

num_classes = 2
model = chargement_model(num_classes)
model.to(device)

#Définir l'optimiseur et le scheduler pour l'entraînement

params = [p for p in model.parameters() if p.requires_grad] 
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005) #Initialisation de l'optimiseur :  taux d'apprentissage (On a pris 0.005 qui est un pas modéré), momentum (permet d'accélérer la convergence et d'éviter les minima locaux)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# Nombre d'epoch pour l'entraînement
num_epochs = 1
# for epoch in range(num_epochs):
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#     lr_scheduler.step()
#     # Évaluation sur l'ensemble de test (uniquement si le dispositif est le GPU)
#     if device.type == 'cuda':
#         evaluate(model, data_loader_test, device=device)


evaluate(model, data_loader_test, device=device)