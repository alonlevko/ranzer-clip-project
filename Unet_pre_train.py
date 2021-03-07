import cpgan_model, cpgan_data, cpgan_tools
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

from collections import defaultdict
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, dataloaders, save_path, num_epochs=25, ):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for batch in dataloaders[phase]:
                inputs = batch["fore"].to(device)
                labels = batch["mask"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs[0], labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.save(model.state_dict(), save_path.format(epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 6
    model = cpgan_model.MyMiniUNet(3, 1, border_zero=True).to(device)
    dataset = "xrays_masks"
    img_dim = 400
    batch_size = 16
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
    # freeze backbone layers
    #for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False
    model_path = "Unets_pre/Unet_{}.pt"
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    dataloaders = {"train": train_loader, "val": val_loader}
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    start_epoch = cpgan_tools.get_last_epoch(model_path)
    total_epoces = 50 - start_epoch
    if start_epoch > 0:
        model.load_state_dict(torch.load(model_path.format(start_epoch), map_location=device))
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=total_epoces, dataloaders=dataloaders, save_path=model_path)
    torch.save(model.state_dict(), model_path + "/Best_Unet_Model.pt")

