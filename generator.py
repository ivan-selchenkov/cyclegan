from torch import nn

from residual_block import ResidualBlock
from utils import conv, deconv


class CycleGenerator(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        self.conv_dim = conv_dim
        self.n_res_blocks = n_res_blocks

        self.encoder_conv1 = conv(3, conv_dim, 2, 2, 1, True)
        self.encoder_conv2 = conv(conv_dim, conv_dim * 2, 2, 2, 1, True)
        self.encoder_conv3 = conv(conv_dim * 2, conv_dim * 4, 2, 2, 1, True)

        self.residual_block_layers = []
        for i in range(n_res_blocks):
            self.residual_block_layers.append(ResidualBlock(conv_dim * 4))

        self.residual_blocks = nn.Sequential(*self.residual_block_layers)

        self.decoder_conv1 = deconv(conv_dim * 4, conv_dim * 2, 2, 2, 1, True)
        self.decoder_conv2 = deconv(conv_dim * 2, conv_dim, 2, 2, 1, True)
        self.decoder_conv3 = deconv(conv_dim, 3, 2, 2, 1, False)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.encoder_conv1(x))
        x = self.relu(self.encoder_conv2(x))
        x = self.relu(self.encoder_conv3(x))

        x = self.residual_blocks(x)

        x = self.relu(self.decoder_conv1(x))
        x = self.relu(self.decoder_conv2(x))
        x = self.tanh(self.decoder_conv3(x))

        return x
