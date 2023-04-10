import os
import numpy as np
import pickle
from PIL import Image

# Función para cargar las imágenes y convertirlas en un arreglo NumPy
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        print(img.size)  # Imprime la forma de la imagen cargada
        # Redimensionar la imagen a 32x32 píxeles con 3 canales de color
        img = img.resize((32, 32))
        # Convertir la imagen en un arreglo NumPy
        img_array = np.array(img)
        # Normalizar los valores de píxeles al rango [0, 1]
        img_array = img_array / 255.0
        images.append(img_array)
    return images

# Cargar las imágenes de todas las carpetas (0 a 5)
data_path = "data"
all_images = []
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    images = load_images_from_folder(folder_path)
    print(f"Se cargaron {len(images)} imágenes de la carpeta {folder_name}")
    all_images.extend(images)

# Convertir las imágenes cargadas en un arreglo NumPy y guardarlas en un archivo pickle
X = np.array(all_images)
with open('features.pkl', 'wb') as f:
    pickle.dump(X, f)

# Comprobar que la matriz tenga la misma longitud (2000 imágenes) para cada clase
for i in range(6):
    indices = np.where(y == i)[0]
    print(f"Cantidad de imágenes de la clase {i}: {len(indices)}")