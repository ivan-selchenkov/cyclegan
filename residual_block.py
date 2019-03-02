from torch import nn
from utils import conv

class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv(conv_dim, conv_dim, 3, stride=1, padding=1, batch_norm=True)
        self.conv2 = conv(conv_dim, conv_dim, 3, stride=1, padding=1, batch_norm=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x + self.conv2(x)

        return x
