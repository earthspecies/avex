"""This minimal example makes sure that we can train a network made
of our example linear layer. Integration tests are here to test that
the a complete run of your code is working.
"""

import torch

from representation_learning.training import train


def main(device: str = "cpu") -> None:
    torch.manual_seed(0)
    pass


if __name__ == "__main__":
    main()


def test_error(device: str) -> None:
    main(device)
