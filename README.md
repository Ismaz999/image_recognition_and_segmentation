# image_recognition_and_segmentation

Ce dépôt contient le code et les fichiers nécessaires pour réaliser un projet de segmentation d'images à l'aide de Mask R-CNN. Le projet est basé sur le dataset PennFudan Pedestrian.

## Introduction

Ce projet utilise l'algorithme Mask R-CNN pour la segmentation d'objets dans des images. L'algorithme Mask R-CNN combine la détection d'objets (bounding boxes) avec la segmentation sémantique des objets (masks), permettant ainsi d'identifier et de localiser précisément les objets d'intérêt dans une image.

## Contenu du projet

Le projet est structuré comme suit :

- `dataset_manager.py`: Ce fichier contient la définition de la classe `PennFudanDataset`, qui gère la préparation des données pour l'entraînement et l'évaluation du modèle.

- `projet_segm.py`: Le script principal qui charge le modèle Mask R-CNN, effectue l'entraînement du modèle et évalue sa performance sur un ensemble de test. Ce script orchestre l'ensemble du processus du projet.

## Configuration et Utilisation

1. Assurez-vous d'avoir les bibliothèques Python nécessaires installées. Vous pouvez les installer en utilisant la commande suivante :

  ```bash
   pip install -r requirements.txt
```

2. Téléchargez le jeu de données PennFudan Pedestrian à partir du lien suivant : [PennFudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/).

3. Dans le fichier `projet_segm.py`, vous pouvez ajuster les paramètres tels que le chemin vers le jeu de données, les transformations de données, le nombre d'époques, etc.

4. Exécutez le script `projet_segm.py` pour effectuer l'entraînement et l'évaluation du modèle Mask R-CNN.
