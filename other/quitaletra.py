import os

data_dir = "test"

for label in range(6):
    label_dir = os.path.join(data_dir, str(label))
    for filename in os.listdir(label_dir):
        if filename.endswith(".jpg"):
            old_name = os.path.join(label_dir, filename)
            new_name = os.path.join(label_dir, ''.join(c for c in filename if c.isdigit()) + ".jpg")
            os.rename(old_name, new_name)
