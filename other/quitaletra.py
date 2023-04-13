import os

data_dir = "test"

for label in range(1):
    label_dir = os.path.join(data_dir, str(label+1))
    for i in range(1, 3001):
        old_name = os.path.join(label_dir, f"H{i}.jpg")
        new_name = os.path.join(label_dir, f"{i}.jpg")
        os.rename(old_name, new_name)