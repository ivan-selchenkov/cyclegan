import os
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn
import torch

import matplotlib.pyplot as plt
import numpy as np

def get_data_loader(image_type, image_dir='data/summer2winter_yosemite',
                    image_size=128, batch_size=16, num_workers=0):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def imshow(img):
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))


def display_dataset(dataloader_X):
    dataiter = iter(dataloader_X)
    images, _ = dataiter.next()
    fig = plt.figure(figsize=(12, 8))
    imshow(make_grid(images))
    plt.show()


def scale(x, feature_range=(-1, 1)):
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    return layers(nn.Conv2d, in_channels, out_channels, kernel_size, stride, padding, batch_norm)


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    return layers(nn.ConvTranspose2d, in_channels, out_channels, kernel_size, stride, padding, batch_norm)


def layers(method, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []

    layer = method(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)

    layers.append(layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def real_mse_loss(D_out):
    return torch.mean((D_out - 1) ** 2)


def fake_mse_loss(D_out):
    return torch.mean(D_out ** 2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    resonstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight * resonstr_loss
