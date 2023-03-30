import pytorchlab
from pytorchlab import nn

net = pytorchlab.net

net.set_stack(
    nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 4)
    )
)

net.fit([[1,0,1,0], [0,1,0,1]], [[1,0,1,0], [0,1,0,1]])