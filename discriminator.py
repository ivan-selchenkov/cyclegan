from torch import nn
from utils import conv

class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # 128x128x3 -> 64x64x64
        self.conv1 = conv(3, conv_dim, 4, stride=2, padding=1, batch_norm=False)
        # 64x64x64 -> 32x32x128
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, stride=2, padding=1, batch_norm=True)
        # 32x32x128 -> 16x16x256
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, stride=2, padding=1, batch_norm=True)
        # 16x16x256 -> 8x8x512
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, stride=2, padding=1, batch_norm=True)

        self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        # define feedforward behavior
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv1(x)

        return x
