import os
import numpy as np
import pickle
import mediapipe as mp
from tensorflow import keras
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Función para cargar las imágenes, detectar las características de las manos y convertirlas en un arreglo NumPy
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
        img_hands = mp_hands.process(img_array)
        if img_hands.multi_hand_landmarks:
            # Obtener las características de las manos
            hand_features = np.array([hand_landmarks.landmark for hand_landmarks in img_hands.multi_hand_landmarks])
            # Aplanar las características de las manos y agregarlas al conjunto de datos
            images.append(hand_features.flatten())
            # Obtener la etiqueta de la imagen a partir del nombre de la carpeta
            label = int(folder.split("\\")[-1])
            labels.append(label)
    return images, labels

# Cargar las imágenes de todas las carpetas (0 a 5) y las etiquetas correspondientes
data_path = "data"
all_images = []
all_labels = []
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    images, labels = load_images_from_folder(folder_path)
    print(f"Se cargaron {len(images)} imágenes de la carpeta {folder_name}")
    all_images.extend(images)
    all_labels.extend(labels)

# Convertir las características de las manos y las etiquetas cargadas en arreglos NumPy y guardarlos en un archivo pickle
X = np.array(all_images)
y = np.array(all_labels)
with open('hand_features.pkl', 'wb') as f:
    pickle.dump(X, f)
    pickle.dump(y, f)

# Comprobar que la matriz tenga la misma longitud (2000 imágenes) para cada clase
for i in range(5):
    indices = np.where(y == i)[0]
    print(f"Cantidad de imágenes de la clase {i}: {len(indices)}")

# Separar los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Modificar la forma de entrada de la primera capa del modelo
model = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(units=6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=6, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"Exactitud en el conjunto de prueba: {test_acc}")
model.save('model.h5')


