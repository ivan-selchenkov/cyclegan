import os
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn
import torch

import matplotlib.pyplot as plt
import numpy as np

import scipy
import scipy.misc


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


def merge_images(sources, targets, batch_size=16):
    """Creates a grid consisting of pairs of columns, where the first column in
        each pair contains images source images and the second column in each pair
        contains images generated by the CycleGAN from the corresponding images in
        the first column.
        """
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    merged = merged.transpose(1, 2, 0)
    return merged


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    x = ((x +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
    return x


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='samples_cyclegan'):
    if torch.cuda.is_available():
        fixed_Y = fixed_Y.to("cuda")
        fixed_X = fixed_X.to("cuda")

    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)

    merged = merge_images(X, fake_Y, batch_size)
    path = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, batch_size)
    path = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))
