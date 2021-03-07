import math
import torch
import cpgan_model, cpgan_data
import os

import matplotlib.pyplot as plt
import numpy as np

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce
def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            masks_indexes = masks[:,y,x] > 0.3
            masks_indexes = masks_indexes.astype(np.uint)
            selected_colors = colors[masks_indexes]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = cpgan_model.MyMiniUNet(3, 1, border_zero=True).to(device)
    model.load_state_dict(torch.load("Unets_pre/Unet_23.pt", map_location=device))
    model.eval()   # Set model to the evaluation mode

    dataset = "xrays_masks"
    img_dim = 400
    batch_size = 8
    # custom dataset from folders containing images
    train_back_dir = 'data/' + dataset + '/train_back/'
    train_fore_dir = 'data/' + dataset + '/train_fore/'
    train_mask_dir = 'data/' + dataset + '/train_mask/'
    val_back_dir = 'data/' + dataset + '/val_back/'
    val_fore_dir = 'data/' + dataset + '/val_fore/'
    val_mask_dir = 'data/' + dataset + '/val_mask/'
    if not(os.path.exists(train_mask_dir)):
        train_mask_dir = None
    if not(os.path.exists(val_mask_dir)):
        val_mask_dir = None # cannot measure ODP in that case
    train_data = cpgan_data.MyCopyPasteDataset(train_fore_dir, train_back_dir, train_mask_dir, post_resize=img_dim, center_crop=False)
    val_data = cpgan_data.MyCopyPasteDataset(val_fore_dir, val_back_dir, val_mask_dir, post_resize=img_dim, center_crop=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Get the first batch
    batch = next(iter(val_loader))
    inputs = batch["fore"].to(device)
    labels = batch["mask"].to(device)

    # Predict
    pred, score = model(inputs)
    print(torch.max(pred))
    print(torch.min(pred))
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    print(torch.max(pred))
    print(torch.min(pred))
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [masks_to_colorimg(x) for x in pred]

    plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
    print("done")