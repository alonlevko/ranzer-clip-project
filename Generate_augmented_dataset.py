import cpgan_data, cpgan_tools
import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt
from cpgan_model import MyMiniUNet
import cv2
device = "cuda"
trans = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((400, 400)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

def Unet_get_model():
    model_path = "Unets_pre/Unet_23.pt"
    model = MyMiniUNet(3, 1, border_zero = True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def masks_to_colorimg(masks):
    colors = np.asarray([255, 0])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 1), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            masks_indexes = masks[:,y,x] > 0.3
            masks_indexes = masks_indexes.astype(np.uint)
            selected_colors = colors[masks_indexes]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


if __name__ == '__main__':
    save_path = "data/Fake_data_unet_mask_copy/"
    back_src = "data/NIH_data/images/"
    fore_src = "data/Competition_data/train/"
    mask_src = "data/Competition_data/train_masks/"
    fores_df = pd.read_csv("data/Competition_data/train.csv")
    train_annot_df = pd.read_csv("data/Competition_data/train_annotations.csv")
    labels = train_annot_df['label'].unique()
    labels_count_dict = {}
    for label in labels:
        loc = fores_df.loc[fores_df[label] == 1].index
        labels_count_dict[label] = len(loc)
    print(labels_count_dict)
    background_filenames = os.listdir(back_src)
    np.random.shuffle(background_filenames)
    back_img_index = 0
    gen_idx = 1
    # make every label that has less than 3000 images have 3000 images
    fake_data_list = []
    model = Unet_get_model()
    model.eval()
    for k, n in labels_count_dict.items():
        if k == "NGT - Incompletely Imaged":
            continue
        for_imgs = fores_df.loc[fores_df[k]==1]
        #for_imgs = train_annot_df.loc[train_annot_df['label']==k]
        for_imgs = np.unique(for_imgs[["StudyInstanceUID"]].to_numpy())
        for_imgs_index = 0
        copies = min(3000, n * 10)
        for i in range(n, 3000):
            # generate and save fake image
            fore_img_path = fore_src + for_imgs[for_imgs_index] + ".jpg"
            back_img_path = back_src + background_filenames[back_img_index]
            mask_img_path = mask_src + for_imgs[for_imgs_index] + ".jpg"
            fore_img_pil = Image.open(fore_img_path)
            fore_img = np.array(fore_img_pil)
            back_img = Image.open(back_img_path)
            fore_img = cv2.cvtColor(fore_img, cv2.COLOR_BGR2RGB)
            fore_img = trans(fore_img)
            back_img = back_img.resize((400, 400))
            fore_img_pil = fore_img_pil.resize((400, 400))
            fore_img = fore_img[None, :, :, :]
            fore_img = fore_img.to(device)
            mask_img, _ = model.forward(fore_img)
            mask_img = mask_img.cpu()
            fore_img = fore_img.cpu()
            mask_img = mask_img[-1, :, :, :]
            mask_img = masks_to_colorimg(mask_img.detach().numpy())
            #mask_img = Image.open(mask_img_path)
            comp_name = for_imgs[for_imgs_index] + "_" + background_filenames[back_img_index]
            for_imgs_index = (for_imgs_index + 1) % len(for_imgs)
            back_img_index = (back_img_index + 1) % len(background_filenames)
            mask_img = Image.fromarray(mask_img[:, :, -1])
            comp_img = Image.composite(fore_img_pil, back_img, mask_img)
            comp_img.save(save_path + comp_name + ".png")
            data_dict = {"StudyInstanceUID": comp_name, "PatientID": "fffffff"}
            for label in labels:
                data_dict[label] = "0"
            data_dict[k] = "1"
            fake_data_list.append(data_dict)
    df = pd.DataFrame(fake_data_list)
    all_images = fores_df.append(df)
    all_images.to_csv(path_or_buf=save_path + "train_f.csv", index=False)