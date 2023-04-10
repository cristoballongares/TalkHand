import cv2
import numpy as np
from tensorflow import keras

# Cargar el modelo entrenado
model = keras.models.load_model('model.h5')

# Capturar la imagen en tiempo real
cap = cv2.VideoCapture(1)
labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

while True:
    # Leer la imagen capturada
    ret, frame = cap.read()

    # Transformar la imagen
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Hacer la predicción utilizando el modelo
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction)
    predicted_letter = labels_dict[predicted_class]

    # Mostrar la predicción al usuario
    cv2.putText(frame, f"Predicción: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Captura de video', frame)

    # Esperar a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()