import os

# Directorio base
base_dir = 'datanew2'

# Recorremos las 5 carpetas numeradas del 0 al 4
for i in range(5):
    dir_path = os.path.join(base_dir, str(i))
    
    # Verificamos si la carpeta existe
    if os.path.exists(dir_path):
        # Recorremos las 1000 fotos numeradas del 0 al 999
        for j in range(1000):
            old_name = os.path.join(dir_path, '{}.jpg'.format(j))
            new_name = os.path.join(dir_path, '{}.jpg'.format(j+1000))
            # Renombramos el archivo
            os.rename(old_name, new_name)
            
            print(f'Archivo {old_name} renombrado a {new_name}')
    else:
        print(f'La carpeta {dir_path} no existe')