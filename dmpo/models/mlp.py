import torch.nn as nn

from torch import Tensor
from typing import Optional, List, Union, Dict, Any

from .. import utils

class MLP(nn.Module):
    """
    Class for constructing multilayer perceptrons
    """
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: List[int],
                 act: Union[str, nn.Module],
                 dropout_prob: Optional[float]=None,
                 init_scale: Optional[float]=None,
                 act_params: Dict[str, Any]={},
                 last_linear: bool=True) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        # Convert activation string to Module
        if isinstance(act, str) and act != 'identity':
            act = utils.ActivationType[act]

        # Create the MLP
        net = []
        prev_dim = in_size
        if not hidden_size is None and len(hidden_size) > 0:
            for hidden in hidden_size:
                # Construct the linear layer
                layer = nn.Linear(prev_dim, hidden)
                net.append(layer)

                # Apply dropout
                if not dropout_prob is None:
                    net.append(nn.Dropout(p=dropout_prob))

                # Append activation
                if act != 'identity':
                    net.append(act(**act_params))

                prev_dim = hidden

        # Apply terminal linear layer
        if last_linear:
            layer = nn.Linear(prev_dim, out_size)
            if not init_scale is None:
                layer.weight.data.normal_(0, init_scale)
                layer.bias.data.fill_(0)
            net.append(layer)

        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        # Automatically reshape input to be (batch_size, input_dim)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        out = self.net(x)

        # Reshape input to original shape
        out = out.reshape(*x_shape[:-1], -1)
        return out