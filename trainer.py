import torch
from typing import List, Tuple


def fit(model: torch.nn.Module, trainX: List[List[float]], trainY: List[List[float]],
        valX: List[List[float]] = None, valY: List[List[float]] = None,
        epochs: int = 100, lr: float = 0.001, batch_size: int = 32) -> Tuple:
    """Train a PyTorch model using the specified data and hyperparameters.

    Args:
        model: PyTorch model to train.
        trainX: List of input data for training.
        trainY: List of output data for training.
        valX: List of input data for validation (optional).
        valY: List of output data for validation (optional).
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size for training.

    Returns:
        A tuple containing training loss history and validation loss history (if validation data was provided).
    """
    # Convert input and output data to PyTorch tensors
    x_train = torch.tensor(trainX, dtype=torch.float32)
    y_train = torch.tensor(trainY, dtype=torch.float32)
    if valX is not None and valY is not None:
        x_val = torch.tensor(valX, dtype=torch.float32)
        y_val = torch.tensor(valY, dtype=torch.float32)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create data loader for training data
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create data loader for validation data (if provided)
    if valX is not None and valY is not None:
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None

    # Train the model
    train_loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
        # Train for one epoch
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.shape[0]  # Multiply by batch size to account for different batch sizes
        epoch_train_loss /= len(train_dataset)
        train_loss_history.append(epoch_train_loss)

        # Evaluate on validation data (if provided)
        if val_loader is not None:
            epoch_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    epoch_val_loss += loss.item() * x_batch.shape[0]
                epoch_val_loss /= len(val_dataset)
                val_loss_history.append(epoch_val_loss)

    if val_loader is not None:
        return train_loss_history, val_loss_history
    else:
        return train_loss_history
