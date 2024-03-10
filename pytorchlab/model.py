import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt


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
        self.best_model_validation = {}
        self.validation_error_history = []
        self.train_error_history = []

        super().__init__()

    def update_validation_history(self, last_validation_error):
        self.validation_error_history.append(last_validation_error)
        if min(self.validation_error_history) == last_validation_error:
            self.best_model_validation = self.state_dict().copy()
        
    def update_train_history(self, last_train_error):
        self.train_error_history.append(last_train_error)
        

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

            loss, current = loss.item(), (batch + 1) * len(X)
            self.update_train_history(loss)
            if batch % 100 == 0:
                print(f"[Training] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def _test(self):
        raise Exception("testing not supported yet!")

    def _validation(self, dataloader, model, loss_fn, device):
        
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss, current = loss.item(), (batch + 1) * len(X)

            self.update_validation_history(loss)
            if batch % 100 == 0:    
                print(f"[Validation] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        

    def plot(self, validation=False):
        # t = range(len(self.train_error_history))
        plt.plot(self.train_error_history)
        if validation:
            plt.plot(self.validation_error_history)

        plt.show()


    def _multi_dimention_list_to_numpy_array(self, lst:list):
        return np.array([np.array(l) for l in lst])
                
    def set_stack(self, stack : nn.Sequential):
        self.stack = stack
    
    def save(self, path : str):
        torch.save(self.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")

    def save_error_hist(self):
        pass
    
    def load(self, path : str):
        self.load_state_dict(torch.load(path))
        print("Loaded model!")
    

    def _create_dataloaders(
            self, 
            trainX, 
            trainY, 
            testX=[], 
            testY=[], 
            validationX = [],
            validationY = [],
            testing = True,
            validation=True,
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
            testX = self._multi_dimention_list_to_numpy_array(trainX)
            testY = self._multi_dimention_list_to_numpy_array(testY)
            test_tensor_x = torch.Tensor(np.array(testX))
            test_tensor_y = torch.Tensor(np.array(testY))
            test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        #Handle validation data
        validation_dataloader = None
        if validation:
            validationX = self._multi_dimention_list_to_numpy_array(validationX)
            validationY = self._multi_dimention_list_to_numpy_array(validationY)
             # Convert arrays to tensors
            validation_tensor_x = torch.Tensor(validationX)
            validation_tensor_y = torch.Tensor(validationY)

            # Create TensorDataset and DataLoader
            validation_dataset = TensorDataset(validation_tensor_x, validation_tensor_y)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
            


        return train_dataloader, test_dataloader, validation_dataloader


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
            validationX = [],
            validationY=[],
            validation=True,
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
        elif not validationX or not validationY:
            validation = False
        

        # Set device automatically
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Handle batch size argument
        # if batch_size != None:
        #     raise Exception("Custom batch size unsupported yet!")
        
        # Create dataloaders:
        train_dataloader, test_dataloader, validation_dataloader = self._create_dataloaders(
            trainX, trainY,
            validationX=validationX, 
            validationY=validationY,
            testing=testing,
            validation=validation,
            batch_size=batch_size)
        
        model = self.to(device)

        # Initialize optimizer:
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Looping throught epochs (training):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train(train_dataloader, loss_function, optimizer, device)
            if testing:
                self._test(test_dataloader, model, loss_function)
            
            if validation:
                self._validation(validation_dataloader, model, loss_function, device)

        print("Done!")

    
    
        

    
    
    
