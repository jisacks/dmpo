from torch.nn import Module
from .mlp import MLP

from typing import Optional

def create_net(net_type: str, in_size: Optional[int]=None, out_size: Optional[int]=None, **kwargs) -> Module:
    if net_type == 'mlp':
        net = MLP(in_size=in_size, out_size=out_size, **kwargs)
    else:
        raise ValueError('Invalid network type {} specified'.format(net_type))
    return net
