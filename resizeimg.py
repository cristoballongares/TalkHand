import os
from PIL import Image

# Ruta de la carpeta que contiene todas las carpetas con las imágenes originales
input_folder = './data'

# Ruta de la carpeta donde se guardarán las imágenes redimensionadas
output_folder = './datanew'

# Tamaño de las imágenes redimensionadas
new_size = (128, 128)

# Iterar sobre todas las carpetas dentro de la carpeta de entrada
for folder_name in os.listdir(input_folder):
    # Ruta de la carpeta actual
    folder_path = os.path.join(input_folder, folder_name)
    # Saltar si no es una carpeta
    if not os.path.isdir(folder_path):
        continue
    # Crear la carpeta de salida si no existe
    output_folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_folder_path, exist_ok=True)
    # Iterar sobre todas las imágenes en la carpeta actual
    for filename in os.listdir(folder_path):
        # Ruta del archivo de imagen actual
        file_path = os.path.join(folder_path, filename)
        # Abrir la imagen y redimensionarla
        with Image.open(file_path) as img:
            img = img.resize(new_size)
            # Guardar la imagen redimensionada en la carpeta de salida con el mismo nombre de archivo
            output_path = os.path.join(output_folder_path, filename)
            img.save(output_path)