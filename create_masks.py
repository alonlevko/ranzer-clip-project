import os
import numpy as np
import shutil
import pandas as pd
from PIL import Image, ImageDraw

import seaborn as sns
# # Creating Train / Val / Test folders (One time use)
root_dir = 'data/Competition_data'
front = '/train_front'
masks = '/train_masks'
os.makedirs(root_dir + front)
os.makedirs(root_dir + masks)

# Creating partitions of the data after shuffeling
src = "data/Competition_data/train" # Folder to copy images from
annotations = pd.read_csv("data/Competition_data/train_annotations.csv")
annotationsFileNames = np.unique(annotations["StudyInstanceUID"].to_numpy())
chosen = annotations.loc[annotations['label']=='CVC - Normal'].index.values[0]

annotationsFileNamesList = [src+'/' + name + '.jpg' for name in annotationsFileNames.tolist()]
print('annotated files images: ', len(annotationsFileNamesList))
# Copy-pasting images
copied_files_names = []
for name in annotationsFileNamesList:
    try:
        shutil.copy(name, root_dir + front)
        copied_files_names.append(name)
    except:
        continue


def get_point(chosen, seg_df):
    points = []
    for i, point in enumerate(seg_df.iloc[chosen,2].split('],')):
        if i==len(seg_df.iloc[chosen,2].split('],'))-1:
            xy = tuple([int(x) for x in point[2:-2].split(', ')])
            points.append(xy)
        else:
            xy = tuple([int(x) for x in point[2:].split(', ')])
            points.append(xy)
    return points


for name in copied_files_names:
    img = Image.open(name)
    image_id = name.split("/")[3].replace(".jpg", "")
    points = []
    for value in annotations.loc[annotations['StudyInstanceUID']==image_id].index.values:
        points.append(get_point(value, annotations))
    mask = Image.new("L", img.size)
    data = img.load()
    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            data[x, y] = 255
    draw = ImageDraw.Draw(mask)
    for p in points:
        draw.line(p, fill=255, width=20)
    mask.save(root_dir + masks + "/" + str(image_id) + ".jpg")



