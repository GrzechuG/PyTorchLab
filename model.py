import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.stack = nn.Sequential(
                # nn.Linear(28*28, 512),
                # nn.ReLU(),
                # nn.Linear(512, 512),
                # nn.ReLU(),
                # nn.Linear(512, 10)
            )


        def forward(self, x):
            x = self.flatten(x)
            logits = self.stack(x)
            return logits


class network(NeuralNetwork):
    def __init__(self):
        self.input_ranges = []
        super().__init__()

    def create_stack(neuron_numbers):
        params = []
        for i in range(len(neuron_numbers)-1):
            if i == 0:
                params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))
            elif i == len(neuron_numbers)-2:
                params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))
            else:
                params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))

        

    # Create model
    def newff(self, input_ranges, network_info, activation_function=nn.ReLU()):
        raise Exception("Unsupported yet!")
        # self.input_ranges = input_ranges
        # self.activation_function = activation_function
        # # Creates simple network neuron number representations
        # neuron_numbers = [len(input_ranges)] + network_info

    
    def _train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _create_dataloaders(
            self, 
            trainX, 
            trainY, 
            testX=[], 
            testY=[], 
            testing = True,
            batch_size=None
    ):
        train_tensor_x = torch.Tensor(np.array(testX)) 
        train_tensor_y = torch.Tensor(np.array(trainY))

        if testing:
            train_tensor_x = torch.Tensor(np.array(testX))
            train_tensor_y = torch.Tensor(np.array(trainY))

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        test_dataloader = None
        if testing:
            test_dataset = TensorDataset(train_tensor_x, train_tensor_y)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        return train_dataloader, test_dataloader

    def fit(
            self, 
            trainX, 
            trainY, 
            testX=[], 
            testY=[], 
            testing = True, 
            epochs=500,
            batch_size=None
    ):

        if not testX or not testY:
            testing = False

        # Create dataloaders:
        train_dataloader, test_dataloader = self._create_dataloaders(trainX, trainY)
        
        
        
        

        # for X, y in test_dataloader:
        #     print(f"Shape of X [N, C, H, W]: {X.shape}")
        #     print(f"Shape of y: {y.shape} {y.dtype}")
        #     break
        # pass

    def train(self, trainX, trainY, epochs=500):
        return self.fit(trainX, trainY, epochs=500)
    

    def set_stack(stack : nn.Sequential):
        self.stack = stack
    
        

    
    
    
