import torch.nn as nn
from activ import Activation

def fill_list(l: list, target_length: int, default):
    if len(l) >= target_length:
        return
    else:
        to_add = target_length - len(l)
        list.extend(l, [default] * to_add)

def repeat(value, times: int):
    return [value] * times

class MLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_units: 'list[int]',
                 output_size: int,
                 input_dropout: float = 0.0,
                 hidden_dropout: 'float | list[float]' = 0.0,
                 output_dropout: float = 0.0,
                 hidden_activ: 'str | list[str]' = "relu",
                 output_activ: str = "relu",
                ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.n_hidden = len(self.hidden_units)
        self.output_size = output_size
        self.input_dropout = input_dropout
        self.hidden_dropout = self.handle_list_attribute(hidden_dropout, 0.0, float, "hidden_dropout")
        self.output_dropout = output_dropout
        self.hidden_activ = self.handle_list_attribute(hidden_activ, None, str, "hidden_activ")
        self.output_activ = output_activ

        # Build the model
        self.layers = nn.ModuleList([])
        # >>> Input Dropout
        if self.input_dropout > 0:
            self.layers.add_module("Input_dropout", nn.Dropout(p = self.input_dropout))
        # >>> Hidden Layers
        in_size = self.input_size
        for i, units in enumerate(self.hidden_units):
            self.layers.add_module("Dense_{}_{}".format(i, units), nn.Linear(in_features=in_size, out_features=units))
            if self.hidden_dropout[i] > 0:
                self.layers.add_module("Dropout_{}".format(i), nn.Dropout(p = self.hidden_dropout[i]))
            if self.hidden_activ[i] != None:
                self.layers.add_module("Activ_{}".format(i), Activation(self.hidden_activ[i]))
            in_size = units
        # >>> Output Layer
        self.layers.add_module("Output_Dense_{}".format(self.output_size), nn.Linear(in_features=in_size, out_features=self.output_size))
        if self.output_dropout > 0:
            self.layers.add_module("Output_Dropout", nn.Dropout(p = self.output_dropout))
        if self.output_activ != None:
            self.layers.add_module("Output_Activ", Activation(self.output_activ))

    def handle_list_attribute(self, attr, defalut, _class, _name):
        if isinstance(attr, list):
            fill_list(attr, self.n_hidden, defalut)
            return attr
        elif isinstance(attr, _class):
            attr = repeat(attr, self.n_hidden)
            return attr
        else:
            raise ValueError("Invalid argument type: {}".format(_name))

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x