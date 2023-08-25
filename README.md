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
2. **Téléchargement des Données :** Téléchargez le jeu de données PennFudan Pedestrian à partir du lien suivant : [PennFudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip).

3. **Clonage du Référentiel PyTorch Vision :** Clonez le référentiel GitHub PyTorch Vision en utilisant le lien suivant : [https://github.com/pytorch/vision.git](https://github.com/pytorch/vision.git).

4. **Ajout des Fichiers de Détection :** Dans le dossier du projet, placez les fichiers du répertoire `vision/references/detection` du référentiel PyTorch Vision. Ces fichiers sont nécessaires pour la détection d'objets.

5. **Ajustement des Paramètres :** Dans le fichier `projet_segm.py`, vous pouvez ajuster les paramètres tels que le chemin vers le jeu de données, les transformations de données, le nombre d'époques, etc.

6. **Résolution des Erreurs :** Si vous rencontrez les erreurs "AssertionError - Torch not compiled with CUDA enabled" et "AttributeError - module torch has no attribute _six", suivez les étapes de résolution mentionnées précédemment.
  Voici les solutions correspondantes :

   - **AssertionError - Torch not compiled with CUDA enabled :** Supprimez toutes les dépendances liées à Torch et CUDA en utilisant les commandes `conda list torch` et `conda list cuda`. Réinstallez-les ensuite avec les commandes suivantes :
     ```bash
     conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
     conda install -c anaconda cudatoolkit
     ```

   - **AttributeError - module torch has no attribute _six :** Remplacez toutes les occurrences de `torch._six.string_classes` par `"str"` dans le fichier `coco_eval.py`.
   
8. **Exécution du Script :** Exécutez le script `projet_segm.py` pour effectuer l'entraînement et l'évaluation du modèle Mask R-CNN.
   
## Résultats

Après avoir exécuté le script d'entraînement et d'évaluation sur un ensemble de test restreint de 50 images, nous avons obtenu les résultats suivants

### Masque de Segmentation

<img src="https://i.imgur.com/SuksJfJ.png" alt="Masque de Segmentation" width="400">

### Évolution de la Perte d'Entraînement

<img src="https://i.imgur.com/E5cetUj.png" alt="Évolution de la Perte d'Entraînement" width="600">

Il est important de noter que nous avons utilisé un ensemble de données d'entraînement restreint comprenant seulement 50 images pour accélérer le processus d'entraînement. Cependant, en utilisant un nombre plus important d'images pour l'entraînement, les résultats pourraient être considérablement améliorés. 

