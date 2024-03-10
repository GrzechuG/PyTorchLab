import pytorchlab
from pytorchlab import nn

net = pytorchlab.net

net.set_stack(
    nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 4)
    )
)

net.fit([[1,0,1,0]]*100, [[0,1,0,1]]*100,validationX=[[1,0,1,0]]*100, validationY=[[0,1,0,1]]*100, batch_size=64)

net.plot(validation=True)


net.save("model.pt")

net.load("model.pt")




print(net.sim([1,0,1,0]))