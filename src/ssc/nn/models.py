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

        if isinstance(self.resnet.kernel_size, int):
            k = self.resnet.kernel_size
        else:
            k = self.resnet.kernel_size[0]
        self.extra_padding_dim = k // 2
        self.up_extra_padding = torch.nn.Parameter(data = torch.rand(size=(self.extra_padding_dim, self.encoding.output_size)),
                                                   requires_grad = True)
        self.bottom_extra_padding = torch.nn.Parameter(data = torch.rand(size=(self.extra_padding_dim, self.encoding.output_size)),
                                                       requires_grad = True)

        # self.layers = torch.nn.Sequential(
        #     self.encoding,
        #     self.resnet,
        #     self.decoding
        # )

    def forward(self, input): # input_shape = (batch_size, seq_len, 2)
        # # 1) Add channel dimension
        # x = torch.unsqueeze(input, dim=1)
        # # 2) Process input
        # x = self.layers(x)
        # # 6) Squeeze All
        # x = torch.squeeze(x)

        batch_size = input.size()[0]
        # 1) Add channel dimension
        x = torch.unsqueeze(input, dim=1)
        # 2) Encoding
        x = self.encoding(x)
        # 3) Add extra padding
        up_extra_padding = self.up_extra_padding[None, None, ...]
        up_extra_padding = up_extra_padding.repeat(batch_size, 1, 1, 1)
        bot_extra_padding = self.bottom_extra_padding[None, None, ...]
        bot_extra_padding = bot_extra_padding.repeat(batch_size, 1, 1, 1)
        x = torch.cat([up_extra_padding, x, bot_extra_padding], dim = 2)
        # 4) Resnet
        x = self.resnet(x)
        # 5) Remove extra padding
        x = x[:,:,self.extra_padding_dim : -self.extra_padding_dim, :]
        # 6) Decoding
        x = self.decoding(x)
        # 7) Squeeze all
        x = torch.squeeze(x)

        return x
