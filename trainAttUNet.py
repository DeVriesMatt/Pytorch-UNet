import argparse
import logging
import os
import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset import BasicDataset

dir_img = './data/IOSTAR/image/train/'   # './augmented_patch/image/'
dir_mask = './data/IOSTAR/GT/train/'  # './augmented_patch/GT/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(degrees=(90, 90)),
                                   transforms.RandomRotation(degrees=(180, 180)),
                                   transforms.RandomRotation(degrees=(270, 270)),
                                   transforms.RandomVerticalFlip(p=1),
                                   transforms.RandomHorizontalFlip(p=1),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='LR_{0}_BS_{1}_SCALE_{2}'.format(lr, batch_size, img_scale))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {0}
        Batch size:      {1}
        Learning rate:   {2}
        Training size:   {3}
        Validation size: {4}
        Checkpoints:     {5}
        Device:          {6}
        Images scaling:  {7}
    '''.format(epochs, batch_size, lr, n_train, n_val, save_cp, device.type, img_scale))

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if 1 > 1 else 'max', patience=2)
    if 1 > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch {0}/{1}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == 3, \
                    'Network has been defined with {0} input channels, \n'\
                    'but loaded images have {1} channels. Please check that \n ' \
                    'the images are loaded correctly.'.format(3, imgs.shape[1])

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if 1 == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if 1 > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if 1 == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoint {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = AttU_Net()
    if isinstance(net, UNet):
        logging.info('Network:\n'
                     '\t{0} input channels\n'
                     '\t{1} output channels (classes)\n'
                     '\t{2} upscaling'.format(net.n_channels, net.n_classes, "Bilinear" if net.bilinear else "Transposed conv"))
    else:
        logging.info('Network:\n'
                     '\t{0} input channels\n'
                     '\t{1} output channels (classes)'.format(3, 1))

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    date = datetime.datetime.now()
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)

        # Saving model information
        model_string = 'MODEL' + date.strftime("%c") + '.pth'
        model_string = model_string.replace(" ", "_")
        model_string = model_string.replace(":", "")
        net_name = net.__class__.__name__
        image_trained = dir_img.replace("/", "_")
        image_trained = image_trained.replace(".", "")

        torch.save(net.state_dict(), net_name + image_trained + model_string)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
