import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer
import numpy as np

class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.stack = nn.Sequential(
                # nn.Linear(28*28, 512),
                # nn.ReLU(),
                # nn.Linear(512, 512),
                # nn.ReLU(),
                # nn.Linear(512, 10)
            )


        def forward(self, x):
            # x = self.flatten(x)
            logits = self.stack(x)
            return logits


class network(NeuralNetwork):
    def __init__(self):
        self.input_ranges = []
        super().__init__()

    def create_stack(neuron_numbers):
        raise Exception("Unsupported yet!")
        # params = []
        # for i in range(len(neuron_numbers)-1):
        #     if i == 0:
        #         params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))
        #     elif i == len(neuron_numbers)-2:
        #         params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))
        #     else:
        #         params.append(nn.Linear(neuron_numbers[i],neuron_numbers[i+1]))

        

    # Create model
    def newff(self, input_ranges, network_info, activation_function=nn.ReLU()):
        raise Exception("Unsupported yet!")
        # self.input_ranges = input_ranges
        # self.activation_function = activation_function
        # # Creates simple network neuron number representations
        # neuron_numbers = [len(input_ranges)] + network_info

    
    def _train(self, dataloader, loss_fn, optimizer, device):
        model = self.to(device)
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


    def _test(self):
        raise Exception("testing not supported yet!")


    def _multi_dimention_list_to_numpy_array(self, lst:list):
        return np.array([np.array(l) for l in lst])
                
    def set_stack(self, stack : nn.Sequential):
        self.stack = stack
    
    def save(self, path : str):
        torch.save(self.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")
    
    def load(self, path : str):
        self.load_state_dict(torch.load(path))
        print("Loaded model!")
    

    def _create_dataloaders(
            self, 
            trainX, 
            trainY, 
            testX=[], 
            testY=[], 
            testing = True,
            batch_size=None
    ):

        # Convert lists to numpy arrays
        trainX = self._multi_dimention_list_to_numpy_array(trainX)
        trainY = self._multi_dimention_list_to_numpy_array(trainY)
        
        # Convert arrays to tensors
        train_tensor_x = torch.Tensor(trainX)
        train_tensor_y = torch.Tensor(trainY)

        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Handle test data
        test_dataloader = None
        if testing:
            test_tensor_x = torch.Tensor(np.array(testX))
            test_tensor_y = torch.Tensor(np.array(testY))
            test_dataset = TensorDataset(train_tensor_x, train_tensor_y)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        return train_dataloader, test_dataloader


    def sim(self, X):
        X = self._multi_dimention_list_to_numpy_array(X)
        
        # Convert arrays to tensors
        train_tensor_x = torch.Tensor(X)
        pred = self(train_tensor_x)
        return pred.tolist()
    

    def fit(
            self, 
            trainX : list, 
            trainY : list, 
            testX=[], 
            testY=[], 
            testing = True, 
            epochs=500,
            batch_size=None,
            loss_function = nn.MSELoss(),
            optimizer_class : Optimizer = torch.optim.AdamW,
            device = "auto",
            learning_rate = 0.001
    ):

        # Initial argument checks:

        # Compare input lists lengths
        assert len(trainX) == len(trainY), f"Length of trainX {len(trainX)} does not match trainY length ({len(trainY)})"

        # Check if testing data is defined
        if not testX or not testY:
            testing = False
        else:
            raise Exception("Automatic test validation not supported yet!")

        # Set device automatically
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Handle batch size argument
        # if batch_size != None:
        #     raise Exception("Custom batch size unsupported yet!")
        
        # Create dataloaders:
        train_dataloader, test_dataloader = self._create_dataloaders(trainX, trainY, batch_size=batch_size)
        model = self.to(device)

        # Initialize optimizer:
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Looping throught epochs (training):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train(train_dataloader, loss_function, optimizer, device)
            if testing:
                self._test(test_dataloader, model, loss_function)
        print("Done!")

    
    
        

    
    
    
