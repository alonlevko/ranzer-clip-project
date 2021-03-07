import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'data/Competition_data'
train = '/train_splited'
train_mask = '/train_splited_mask'
val = '/val_splited'
val_mask = '/val_splited_mask'
os.makedirs(root_dir + train)
os.makedirs(root_dir + val)
os.makedirs(root_dir + train_mask)
os.makedirs(root_dir + val_mask)
# Creating partitions of the data after shuffeling
src = "data/Competition_data/train_front" # Folder to copy images from
src_masks = "data/Competition_data/train_masks"
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8)])


train_FileNames_list = [src+'/' + name for name in train_FileNames.tolist()]
train_mask_FileNames_list = [src_masks+'/' + name for name in train_FileNames.tolist()]
val_FileNames_list = [src+'/' + name for name in val_FileNames.tolist()]
val_mask_FileNames_list = [src_masks+'/' + name for name in val_FileNames.tolist()]
print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))

# Copy-pasting images
for name in train_FileNames_list:
    shutil.copy(name, root_dir + train)

for name in val_FileNames_list:
    shutil.copy(name, root_dir + val)

for name in train_mask_FileNames_list:
    shutil.copy(name, root_dir + train_mask)

for name in val_mask_FileNames_list:
    shutil.copy(name, root_dir + val_mask)