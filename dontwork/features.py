import os
import numpy as np
import pickle
import mediapipe as mp
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2

# Crear un objeto de detección de manos de Mediapipe
mp_hands = mp.solutions.hands

# Función para detectar y extraer las características de las manos en una imagen
def extract_hand_features(img):
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detección de manos con Mediapipe Hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        results = hands.process(img_gray)
    # Verificar si se detectaron manos en la imagen
    if results.multi_hand_landmarks is None:
        # Si no se detectaron manos, devolver un arreglo vacío
        return np.array([])
    else:
        # Si se detectaron manos, extraer las características de las manos
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_features = np.zeros((21, 3))
        for i, landmark in enumerate(hand_landmarks.landmark):
            hand_features[i] = [landmark.x, landmark.y, landmark.z]
        return hand_features.flatten()

# Función para cargar las imágenes y convertirlas en un arreglo NumPy, utilizando Mediapipe Hands para extraer las características de las manos
def load_images_from_folder(folder):
    images = []
    labels = []
    # Recorrer los subdirectorios en el directorio "folder"
    for subfolder_name in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder_name)
        # Verificar si es un directorio
        if os.path.isdir(subfolder_path):
            # Recorrer las imágenes en el subdirectorio
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                # Redimensionar la imagen a 128x128 píxeles con 3 canales de color
                img = cv2.resize(img, (128, 128))
                # Extraer las características de las manos con Mediapipe Hands
                hand_features = extract_hand_features(img)
                # Verificar si se detectaron manos en la imagen
                if hand_features.size == 0:
                    # Si no se detectaron manos, saltar la imagen
                    continue
                # Normalizar los valores de las características al rango [0, 1]
                hand_features = hand_features / 1000.0
                # Añadir las características de las manos y la etiqueta de la imagen a las listas correspondientes
                images.append(hand_features)
                # Obtener la etiqueta de la imagen a partir del nombre del subdirectorio
                label = int(subfolder_name)
                labels.append(label)
    return images, labels

images, labels = load_images_from_folder("data")


with open("hand_features.pkl", "wb") as f:
    pickle.dump({"images": images, "labels": labels}, f)
