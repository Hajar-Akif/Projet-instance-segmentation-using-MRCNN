import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_training_history():
    """Visualise l'historique d'entraînement du modèle"""
    # Vérifier si le fichier existe
    if not os.path.exists('bird_segmentation_history.npy'):
        print("Erreur : Le fichier 'bird_segmentation_history.npy' est introuvable.")
        return

    # Charger l'historique d'entraînement
    history = np.load('bird_segmentation_history.npy', allow_pickle=True).item()
    
    # Configurer le style des graphiques
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Graphique de la précision
    plt.subplot(2, 2, 1)
    plt.plot(history['accuracy'], label='Entraînement', marker='o')
    plt.plot(history['val_accuracy'], label='Validation', marker='o')
    plt.title("Évolution de la Précision")
    plt.xlabel("Époque")
    plt.ylabel("Précision")
    plt.legend()
    plt.grid(True)
    
    # 2. Graphique de la perte
    plt.subplot(2, 2, 2)
    plt.plot(history['loss'], label='Entraînement', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='o')
    plt.title("Évolution de la Perte")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    
    # 3. Graphique du taux d'apprentissage
    if 'learning_rate' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rate'], marker='o', color='green')
        plt.title("Évolution du Taux d'Apprentissage")
        plt.xlabel("Époque")
        plt.ylabel("Taux d'apprentissage")
        plt.yscale('log')
        plt.grid(True)
    
    # 4. Tableau récapitulatif des meilleures performances
    plt.subplot(2, 2, 4)
    best_val_acc = max(history['val_accuracy'])
    best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) + 1
    best_val_loss = min(history['val_loss'])
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1

    plt.axis('off')
    plt.text(0.1, 0.8, "Meilleures Performances:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.6, f"Précision max (validation): {best_val_acc:.4f} (époque {best_val_acc_epoch})")
    plt.text(0.1, 0.4, f"Perte min (validation): {best_val_loss:.4f} (époque {best_val_loss_epoch})")

    # Ajuster la mise en page et afficher le graphique
    plt.tight_layout()
    plt.show()


def plot_learning_curves():
    """Crée un graphique détaillé des courbes d'apprentissage"""
    if not os.path.exists('bird_segmentation_history.npy'):
        print("Erreur : Le fichier 'bird_segmentation_history.npy' est introuvable.")
        return

    history = np.load('bird_segmentation_history.npy', allow_pickle=True).item()
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Courbe de précision
    ax1.plot(history['accuracy'], 'b-', label='Entraînement')
    ax1.plot(history['val_accuracy'], 'r-', label='Validation')
    ax1.set_title("Courbe de Précision")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Précision")
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Courbe de perte
    ax2.plot(history['loss'], 'b-', label='Entraînement')
    ax2.plot(history['val_loss'], 'r-', label='Validation')
    ax2.set_title("Courbe de Perte")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Perte")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_training_metrics():
    """Analyse et affiche les statistiques d'entraînement"""
    if not os.path.exists('bird_segmentation_history.npy'):
        print("Erreur : Le fichier 'bird_segmentation_history.npy' est introuvable.")
        return

    history = np.load('bird_segmentation_history.npy', allow_pickle=True).item()
    
    metrics = {
        'Précision finale': history['accuracy'][-1],
        'Précision de validation finale': history['val_accuracy'][-1],
        'Perte finale': history['loss'][-1],
        'Perte de validation finale': history['val_loss'][-1],
        'Meilleure précision': max(history['accuracy']),
        'Meilleure précision de validation': max(history['val_accuracy']),
        'Meilleure perte': min(history['loss']),
        'Meilleure perte de validation': min(history['val_loss'])
    }
    
    print("\n=== Statistiques d'entraînement ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Analyse de la convergence ===")
    val_loss = history['val_loss']
    patience = 3
    min_improvement = 0.001

    convergence_epoch = 0
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch, loss in enumerate(val_loss):
        if loss < best_loss - min_improvement:
            best_loss = loss
            no_improvement_count = 0
            convergence_epoch = epoch
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            break

    print(f"Convergence atteinte à l'époque: {convergence_epoch + 1}")
    print(f"Meilleure perte de validation: {best_loss:.4f}")


if __name__ == "__main__":
    try:
        print("Génération des graphiques d'entraînement...")
        plot_training_history()
        print("Génération des courbes d'apprentissage détaillées...")
        plot_learning_curves()
        print("Analyse des métriques d'entraînement...")
        analyze_training_metrics()
        print("\nTous les graphiques ont été générés avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'exécution du script: {e}")