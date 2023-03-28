class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
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
        self.input_ranges = input_ranges
        self.activation_function = activation_function
        # Creates simple network neuron number representations
        neuron_numbers = [len(input_ranges)] + network_info
        

    
    
    
