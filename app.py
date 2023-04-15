import cv2
import mediapipe as mp
import pickle
import numpy as np

# Cargar el modelo guardado
with open('my_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Crear un diccionario para mapear las etiquetas a las letras
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N'}

# Inicializar la detección de manos de Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

# Inicializar la captura de video con OpenCV
cap = cv2.VideoCapture(1)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    
    # Convertir la imagen de BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detección de manos con Mediapipe
    results = hands.process(image)
    
    # Si se detectó al menos una mano
    if results.multi_hand_landmarks:
        # Obtener las coordenadas de los puntos de referencia de la mano
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Convertir las coordenadas de los puntos de referencia en características
        features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
        
        # Hacer una predicción con el modelo entrenado
        try:
            label_id = clf.predict([features])[0]
            # Obtener la letra correspondiente a la etiqueta predicha
            label = labels_dict[label_id]
        except:
            label = "Seña desconocida"
        
        # Mostrar la letra en la pantalla
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Mostrar el fotograma en la pantalla
    cv2.imshow('TalkHand', frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()