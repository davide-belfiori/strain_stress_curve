import torch.nn as nn

class Activation(nn.Module):
  def __init__(self, activ_type: str = None) -> None:
    super().__init__()
    self.activ_type = activ_type
    if self.activ_type == None:
      self.activ = nn.Identity()
    elif self.activ_type == "relu":
      self.activ = nn.ReLU()
    elif self.activ_type == "sigmoid":
      self.activ = nn.Sigmoid()
    elif self.activ_type == "tanh":
      self.activ = nn.Tanh()
    elif self.activ_type == "gelu":
      self.activ = nn.GELU()
    else:
      raise ValueError("Invalid activation type \"{activ_type}\"".format(activ_type = self.activ_type))
  
  def forward(self, input):
    return self.activ(input)