# 🚗🚲 Car vs Bike Image Classifier

Ce projet est un modèle d'Intelligence Artificielle basé sur l'apprentissage profond (Deep Learning) permettant de classifier automatiquement des images de **Voitures** et de **Vélos**. 

Il utilise la technique du **Transfer Learning** avec le modèle pré-entraîné `MobileNetV2` via TensorFlow/Keras, ce qui permet d'obtenir d'excellentes performances tout en restant léger et rapide.

## 📊 Le Dataset
Le modèle a été entraîné sur un ensemble de données parfaitement équilibré contenant **4 000 images** au total :
- **2 000** images de vélos (Bike)
- **2 000** images de voitures (Car)

Les images sont automatiquement redimensionnées en `224x224` pixels pour correspondre à l'entrée attendue par l'architecture MobileNetV2.

## 🧠 Architecture du Modèle
Le réseau de neurones est construit avec **TensorFlow/Keras** :
1. **Modèle de base :** `MobileNetV2` (pré-entraîné sur ImageNet, avec les couches de sortie retirées).
2. **GlobalAveragePooling2D :** Réduction de la dimensionnalité spatiale.
3. **Couches Denses & Dropout :** Couches de classification finalisant le réseau tout en évitant le surapprentissage (overfitting).
4. **Optimiseur :** `Adam`
5. **Callbacks utilisés :** `EarlyStopping`, `ReduceLROnPlateau` et `ModelCheckpoint` pour optimiser l'apprentissage et sauvegarder le meilleur modèle.

## 🛠️ Prérequis et Installation
Pour faire tourner ce notebook localement, vous aurez besoin de Python et des bibliothèques suivantes :

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
