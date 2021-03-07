import torch
from PIL import Image
from torchvision import transforms
import os

image_name = "1.2.826.0.1.3680043.8.498.10000428974990117276582711948006105617.jpg"
root_dir = "Competition_data\\resized_images"
cwd = os.getcwd()
#path = r"C:\Projects\Finals_project\data\Competition_data\resized_images\1.2.826.0.1.3680043.8.498.10000428974990117276582711948006105617.jpg"
path = r"C:\Projects\Finals_project\data\xrays\train_back\00000100_000.png"
path2 = r"C:\Projects\Finals_project\data\xrays\train_back\00000001_001.png"

img1 = Image.open(path)
img2 = Image.open(path2)
if img1.mode == 'RGBA':
    img1 = img1.convert("L")
sizes = [400, 300, 250, 200]
for img_size in sizes:
    resizer = transforms.Resize((img_size, img_size))
    imgt = resizer(img1)
    imgt.save(root_dir + "\\img" + str(img_size) + ".jpg")

# pick the image size of 200 on 200