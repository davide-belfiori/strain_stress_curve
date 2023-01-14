import torch.nn as nn
from torch import flatten
from ssc.nn.activ import Activation
from ssc.nn.mlp import MLP

# ---------------
# --- MODULES ---
# ---------------

class Conv2DBlock(nn.Module):

  def __init__(self, 
              in_channels: int, 
              out_channels: int, 
              kernel_size: 'int | tuple' = 3, 
              activ_type: str = None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.activ_type = activ_type
    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
    self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels= self.out_channels, kernel_size=self.kernel_size, padding="same")
    self.activ = Activation(activ_type=self.activ_type)

  def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    return self.activ(x)

class Conv2DResBlock(Conv2DBlock):

  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               kernel_size: 'int | tuple' = 3, 
               activ_type: str = None, 
               drop_rate: float = 0.0):
    super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, activ_type=activ_type)
    self.drop_rate = drop_rate
    self.norm1 = nn.BatchNorm2d(num_features=self.out_channels)
    self.norm2 = nn.BatchNorm2d(num_features=self.out_channels)
    self.res_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
    self.dropout = nn.Dropout2d(p=self.drop_rate)

  def forward(self, input):
    x = self.conv1(input)
    x = self.norm1(x)
    x = self.activ(x)
    x = self.conv2(x)
    x = self.norm2(x)
    res = self.res_conv(input)
    x = x + res
    x = self.activ(x)
    return self.dropout(x)

# --------------
# --- MODELS ---
# --------------

class Conv2DResnet(nn.Module):

  def __init__(self, 
               in_channels: int, 
               filters: list, 
               kernel_size: 'int | tuple' = 3, 
               block_activ_type: str = "relu",
               block_drop_rate: float = 0.0,
               preprocessing = None,
               mlp: MLP = None) -> None:
    super().__init__()
    self.in_channels = in_channels
    self.filters = filters
    self.kernel_size = kernel_size
    self.block_activ_type = block_activ_type
    self.block_drop_rate = block_drop_rate
    self.preprocessing = preprocessing
    self.mlp = mlp
    self.blocks = nn.ModuleList([])

    if self.preprocessing != None:
      in_c = self.preprocessing.out_channels
    else:
      in_c = self.in_channels

    for block_filters in self.filters:
      res_block = Conv2DResBlock(in_channels = in_c,
                                  out_channels = block_filters,
                                  kernel_size = self.kernel_size,
                                  activ_type = self.block_activ_type,
                                  drop_rate = self.block_drop_rate)
      in_c = block_filters
      self.blocks.append(res_block)

  def forward(self, input):
    x = input

    if self.preprocessing != None:
      x = self.preprocessing(input)

    for block in self.blocks:
      x = block(x)

    if self.mlp != None:
      x = flatten(x, start_dim = 1)
      return self.mlp(x)

    return x