import torch
from pytorch_lightning.loggers import TensorBoardLogger

from .experiment_utils import create_model, create_optim
from .. import utils

from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

class Experiment():
    """
    Base class for all experiments
    """
    def __init__(self,
                 model_config: Dict[str, Any],
                 optim_config: Dict[str, Any],
                 n_epochs: int=1,
                 log_folder: str='logs',
                 exp_name: str='experiment',
                 dtype: str='float',
                 device: str='cuda'):
        """
        :param model_config: Configuration for model
        :param optim_config: Configuration for optimizer
        :param n_epochs: # of epochs to run
        :param n_gpus: # of GPUs to utilize
        :param log_folder: Folder to use for logging
        :param exp_name: Name of experiment
        :param dtype: PyTorch datatype to use
        :param device: PyTorch device to use
        :param val_every: # of epochs per which we run the validation step
        :param max_grad_norm: Maximum gradient norm clip
        :param lr_scheduler_config: Configuration for learning rate scheduler (optional)
        """
        self.n_epochs = n_epochs

        # Set tensor args
        self.dtype = utils.TorchDtype[dtype]
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.tensor_args = {'device':self.device, 'dtype':self.dtype}

        # Create the model
        self.model = create_model(tensor_args=self.tensor_args, **model_config)
        self.model.to(**self.tensor_args)

        # Create the logger
        self.logger = TensorBoardLogger(log_folder, exp_name)

        # Get the optimizers and learning rate schedulers
        self.optim, self.optim_args = create_optim(optim_config)
