import cv2
import os
import numpy as np
import pickle
import mediapipe as mp

# Cargar el modelo de detección de manos de Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Ruta a la carpeta con las imágenes
data_dir = "data"

num_images = 6000

features = []
labels = []

for label in range(8):
    label_dir = os.path.join(data_dir, str(label))
    
    for i in range(num_images):
        img_path = os.path.join(label_dir, f"{i+1}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error al leer la imagen: {img_path}")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        with mp_hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                continue
            
            annotated_image = img.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_features = []
            for lm in hand_landmarks.landmark:
                hand_features.append(lm.x)
                hand_features.append(lm.y)
                hand_features.append(lm.z)
            features.append(hand_features)
            labels.append(label)
            
            # Guardamos la imagen con los puntos de referencia dibujados
            cv2.imwrite(f"annotated/{label}_{i+1}.jpg", annotated_image)
            
# Guardamos las características y etiquetas en archivos pickle
with open("features.pkl", "wb") as f:
    pickle.dump(features, f)
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)