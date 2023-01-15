import torch
from ssc.nn.cnn import Conv2DResnet
from ssc.nn.activ import Activation
from ssc.nn.mlp import MLP

class StrainStressNetwork(torch.nn.Module):

    def __init__(self,
                 encoding: MLP,
                 resnet: Conv2DResnet,
                 decoding: MLP) -> None:
        super().__init__()
        self.encoding = encoding
        self.resnet = resnet
        self.decoding = decoding

        self.layers = torch.nn.Sequential(
            self.encoding,
            self.resnet,
            self.decoding
        )

    def forward(self, input): # input_shape = (batch_size, seq_len, 2)
        # 1) Add channel dimension
        x = torch.unsqueeze(input, dim=1)
        # 2) Process input
        x = self.layers(x)
        # 6) Squeeze All
        x = torch.squeeze(x)

        return x
