import os
from PIL import Image

root_dir = "test"

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for idx, img_name in enumerate(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_img_name = f"{idx+3001}.jpg"
                new_img_path = os.path.join(folder_path, new_img_name)
                img.save(new_img_path)
