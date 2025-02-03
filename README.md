# PyTorchLab
Easy to use PyTorch training framework with a single fit method

## Features
- Simple API for training PyTorch models
- Automatic device selection (CPU/GPU/MPS)
- Built-in validation and testing support
- Error history tracking and visualization
- Model checkpointing (save best model)
- Memory optimization for large datasets

## Installation
```bash
pip install git+https://github.com/GrzechuG/PyTorchLab.git
```

## Requirements
- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## Quick Start
```python
import pytorchlab
from pytorchlab import nn

# Create network
net = pytorchlab.net

# Define architecture
net.set_stack(
    nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 4)
    )
)

# Prepare data
X = [[1,0,1,0]] * 100  # input data
y = [[0,1,0,1]] * 100  # target data

# Train model
net.fit(
    trainX=X, 
    trainY=y,
    validationX=X,  # optional validation data
    validationY=y,
    batch_size=64,
    epochs=500
)

# Visualize training history
net.plot(validation=True)

# Make predictions
predictions = net.sim([1,0,1,0])

# Save model
net.save("model.pt")
```

## API Reference

### Network Creation
```python
net = pytorchlab.net
net.set_stack(nn.Sequential(...))
```

### Training
```python
net.fit(
    trainX: List[List[float]],          # Training input data
    trainY: List[List[float]],          # Training target data
    validationX: List[List[float]] = [], # Validation input data
    validationY: List[List[float]] = [], # Validation target data
    batch_size: int = None,             # Batch size for training
    epochs: int = 500,                  # Number of training epochs
    loss_function = nn.MSELoss(),       # Loss function
    optimizer_class = torch.optim.AdamW, # Optimizer
    device: str = "auto",               # Device to use (auto/cuda/cpu/mps)
    learning_rate: float = 0.001        # Learning rate
)
```

### Model Management
```python
net.save("model.pt")                    # Save model
net.save_best_validation("best.pt")     # Save best model based on validation
net.load("model.pt")                    # Load model
net.save_error_hist("history.json")     # Save training history
```

### Inference
```python
predictions = net.sim(input_data)       # Make predictions
```

### Memory Management
```python
net.cleanup()                           # Clean GPU/CPU memory
```

## Advanced Features

### GPU Optimization
The framework automatically handles:
- Device selection (CPU/GPU/MPS)
- Memory pinning for faster GPU transfer
- Multi-worker data loading
- Automatic memory cleanup

### Validation
- Automatic tracking of best model
- Validation error history
- Early model saving based on validation performance

## License
GPL 2.0
