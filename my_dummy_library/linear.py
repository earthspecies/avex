"""This is a dummy library containing an MLP definition
to showcase how to test your library. In practice, this
'my_dummy_library' should be the core of your project
or library! This means that this name should also be
replaced in .github/workflows/pythonapp.yml

Header info specific to ESP
"""

import torch
import torch.nn as nn


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_size=40, n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self: torch.nn.Module,
        n_neurons: int,
        input_size: int,
        bias: bool = True,
        max_norm: float = 0.0,
        combine_dims: bool = False,
    ) -> None:
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm != 0.0:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)

        return wx
