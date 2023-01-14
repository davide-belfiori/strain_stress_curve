import torch
from cnn import Conv2DResnet
from activ import Activation

class StrainStressNetwork(torch.nn.Module):

    def __init__(self,
                 resnet: Conv2DResnet,
                 embed_dim: int = 2,
                 out_activ: str = "relu") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.resnet = resnet
        self.out_activ = out_activ

        self.embed = torch.nn.Linear(in_features = 2, out_features = self.embed_dim)
        self.decode = torch.nn.Linear(in_features = self.embed_dim, out_features = 1)
        self.activ = Activation(self.out_activ)

    def forward(self, input): # input_shape = (batch_size, seq_len, 2)
        # 1) Add channel dimension
        x = torch.unsqueeze(input, dim=1)
        # 2) Coordinates Embedding
        x = self.embed(x)
        # 3) ResNet
        x = self.resnet(x)
        # 4) Decoding
        x = self.decode(x)
        # 5) Output Activation
        x = self.activ(x)
        # 6) Squeeze All
        x = torch.squeeze(x)

        return x
