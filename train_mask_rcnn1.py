import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

# Fonction pour charger et prétraiter une image
def load_and_preprocess_image(image_path):
    """Charge et prétraite une seule image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (640, 480))  # Utilisez la taille appropriée pour votre modèle
    image = image / 255.0  # Normalisation
    return image

# Fonction pour charger les images et les masques de validation
def load_data(images_dir, masks_dir, max_images=10):
    """Charge les images et leurs masques à partir des répertoires donnés"""
    image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir)]
    mask_paths = [os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)]
    
    # Charger les images et les masques
    X = []
    y = []
    for i in range(min(max_images, len(image_paths))):
        image = load_and_preprocess_image(image_paths[i])
        mask = load_and_preprocess_image(mask_paths[i])
        X.append(image)
        y.append(mask)
    return np.array(X), np.array(y)

# Fonction pour visualiser les prédictions
def visualize_prediction(image, mask, prediction, save_path=None):
    """Visualise l'image originale, le masque réel et la prédiction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title('Image Originale')
    axes[0].axis('off') 
    # Masque réel (classe oiseau)
    axes[1].imshow(mask[..., 1], cmap='gray')  # Supposons que le masque est binaire
    axes[1].set_title('Masque Réel')
    axes[1].axis('off') 
    # Prédiction (classe oiseau)
    axes[2].imshow(prediction[..., 1], cmap='gray')  # Prédiction binaire
    axes[2].set_title('Prédiction')
    axes[2].axis('off') 
    
    plt.tight_layout() 
    if save_path: 
        plt.savefig(save_path)
    else:
        plt.show()

# Fonction pour évaluer le modèle
def evaluate_model():
    """Évalue le modèle sur le jeu de validation"""
    # Charger le modèle avec les objets personnalisés (si nécessaire)
    custom_objects = {'ResizeLayer': None}  # Ajustez cela si vous avez des couches personnalisées
    try:
        model = load_model('bird_segmentation_best.keras', custom_objects=custom_objects)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return

    # Charger les données de validation
    images_dir = 'path_to_val_images'  # Chemin vers vos images de validation
    masks_dir = 'path_to_val_masks'    # Chemin vers vos masques de validation
    X_val, y_val = load_data(images_dir, masks_dir, max_images=10)  # Charger un sous-ensemble d'images pour la démo
    X_val = X_val / 255.0  # Normaliser les images

    # Faire des prédictions
    predictions = model.predict(X_val) 
    
    # Initialiser les listes pour les scores IoU et Dice
    iou_scores = []
    dice_scores = []

    for i in range(len(X_val)):
        # Calculer IoU pour la classe oiseau
        true_mask = y_val[i, ..., 1]  # Masque réel pour la classe oiseau
        pred_mask = predictions[i, ..., 1] > 0.5  # Prédiction binaire pour la classe oiseau

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / (union + 1e-7)  # Ajouter une petite valeur pour éviter la division par zéro

        # Calculer Dice score
        dice = (2 * intersection) / (true_mask.sum() + pred_mask.sum() + 1e-7)

        iou_scores.append(iou)
        dice_scores.append(dice)

        # Visualiser et sauvegarder les résultats
        visualize_prediction(
            X_val[i],
            y_val[i],
            predictions[i],
            save_path=f'prediction_results_{i}.png'
        )

    # Afficher les métriques moyennes
    print(f"IoU moyen: {np.mean(iou_scores):.4f}")
    print(f"Dice Score moyen: {np.mean(dice_scores):.4f}")

# Exécuter l'évaluation du modèle
if __name__ == "__main__":
    evaluate_model()