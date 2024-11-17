from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mrcnn.utils import Dataset
from PIL import Image
import os

# Chemin vers les annotations (à adapter selon ton système)
annotation_file = 'C:/Users/dell/Desktop/dataset/annotations_trainval2017/instances_train2017.json'

# Charger les annotations
coco = COCO(annotation_file)

# Liste des catégories disponibles (on s'intéresse aux oiseaux)
categories = coco.loadCats(coco.getCatIds())
bird_names = [bird['name'] for bird in categories]
print("Catégories disponibles :", bird_names)

# Obtenir les IDs pour la catégorie spécifique 'bird'
bird_ids = coco.getCatIds(catNms=['bird'])
image_ids = coco.getImgIds(catIds=bird_ids)

# Charger les images associées
images = coco.loadImgs(image_ids)
print(f"Nombre d'images contenant des oiseaux : {len(images)}")

# Charger et afficher une image avec ses annotations
if len(image_ids) > 0:
    image_id = image_ids[0]  # Utiliser la première image contenant des oiseaux
    image_info = coco.loadImgs([image_id])[0]
    image_path = f'C:/Users/dell/Desktop/dataset/train2017/{image_info["file_name"]}'

    # Vérification si l'image existe
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"L'image {image_path} n'a pas pu être chargée.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Charger les annotations pour cette image
        annotation_ids = coco.getAnnIds(imgIds=[image_id], catIds=bird_ids)
        annotations = coco.loadAnns(annotation_ids)

        # Dessiner les masques sur l'image
        for ann in annotations:
            mask = coco.annToMask(ann)
            image[mask == 1] = [255, 0, 0]  # Colorer les masques en rouge

        # Afficher l'image avec les annotations
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement de l'image : {e}")