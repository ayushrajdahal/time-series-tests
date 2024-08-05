import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from XLSTM_original import xLSTM

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to create sequences for time series data
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# List of dataset files
xlsx_files = [
    "Solar station site 1 (Nominal capacity-50MW).xlsx",
    "Solar station site 2 (Nominal capacity-130MW).xlsx",
    "Solar station site 3 (Nominal capacity-30MW).xlsx",
    "Solar station site 4 (Nominal capacity-130MW).xlsx",
    "Solar station site 5 (Nominal capacity-110MW).xlsx",
    "Solar station site 6 (Nominal capacity-35MW).xlsx",
    "Solar station site 7 (Nominal capacity-30MW).xlsx",
    "Solar station site 8 (Nominal capacity-30MW).xlsx",
    "Wind farm site 1 (Nominal capacity-99MW).xlsx",
    "Wind farm site 2 (Nominal capacity-200MW).xlsx",
    "Wind farm site 3 (Nominal capacity-99MW).xlsx",
    "Wind farm site 4 (Nominal capacity-66MW).xlsx",
    "Wind farm site 5 (Nominal capacity-36MW).xlsx",
    "Wind farm site 6 (Nominal capacity-96MW).xlsx",
]

for site_number, file_name in enumerate(xlsx_files, 1):
    print(f"Processing file: {file_name}")
    data = pd.read_excel('../datasets/' + file_name)
    data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'])
    data.set_index('Time(year-month-day h:m:s)', inplace=True)
    data.columns = data.columns.str.strip()
    data.ffill(inplace=True)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    
    n_steps = 72
    X, y = create_sequences(data_scaled.values, n_steps)
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Define the layer types for the xLSTM model
    layer_types = ['s', 'm', 's']  # Example: 3 layers with types sLSTM, mLSTM, sLSTM

    # Instantiate the model
    input_size = X_train.shape[2]
    hidden_size = 40
    num_heads = 4
    output_size = 1

    model = xLSTM(input_size, hidden_size, num_heads, layer_types).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training the model
    num_epochs = 25
    batch_size = 100

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate training and validation loss
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train.unsqueeze(1)).item()
            val_loss = criterion(model(X_test), y_test.unsqueeze(1)).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plotting training and validation loss
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Test')
    plt.title('Training vs Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy(), predictions))
    mae = mean_absolute_error(y_test.cpu().numpy(), predictions)
    r2 = r2_score(y_test.cpu().numpy(), predictions)

    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')