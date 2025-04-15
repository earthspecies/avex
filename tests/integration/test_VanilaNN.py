"""This minimal example makes sure that we can train a network made
of our example linear layer. Integration tests are here to test that
the a complete run of your code is working.
"""

import torch

from my_dummy_library.nnet import VanillaNN

INPUT_SIZE = 64
EPOCH = 50


def main(device: str = "cpu") -> None:
    torch.manual_seed(0)
    x = torch.rand(4, INPUT_SIZE)

    nnet = VanillaNN(input_size=INPUT_SIZE, dnn_neurons=INPUT_SIZE)
    opt = torch.optim.Adam(nnet.parameters(), lr=0.0005)

    for _i in range(EPOCH):
        out = nnet(x)
        loss = torch.nn.MSELoss()(out, x)
        loss.backward()
        opt.step()

    assert loss < 0.15


if __name__ == "__main__":
    main()


def test_error(device: str) -> None:
    main(device)
