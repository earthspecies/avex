"""This is a dummy library containing an MLP definition
to showcase how to test your library. In practice, this
'my_dummy_library' should be the core of your project
or library! This means that this name should also be
replaced in .github/workflows/pythonapp.yml

Header info specific to ESP
"""

import torch
from torch import nn

from my_dummy_library.linear import Linear


class VanillaNN(nn.Module):
    """A simple vanilla Neural Network.

    Arguments
    ---------
    input_size : int
        Expected feature dimension of the input tensors.
    activation : torch class
        A class used for constructing the activation layers.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = VanillaNN(input_size=60)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self: torch.nn.Module,
        input_size: int,
        activation: torch.nn.Module = torch.nn.LeakyReLU,
        dnn_neurons: int = 512,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Linear(dnn_neurons, input_size=input_size),
            activation(),
            Linear(dnn_neurons, input_size=dnn_neurons),
            activation(),
        )

    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Simply pass x throughout the neural network

        Arguments
        ---------
        x : torch.Tensor
            Input to the model.

        Returns
        -------
        out : torch.Tensor
            The output of the model.
        """

        return self.model(x)
