import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from xlstm import xLSTMLMModel, xLSTMLMModelConfig
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# xLSTM configuration
xlstm_cfg = """ 
vocab_size: 50304
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 256
num_blocks: 7
embedding_dim: 128
slstm_at: [1]
"""

xlsx_files = [
    "Wind farm site 1 (Nominal capacity-99MW).xlsx",
    "Wind farm site 2 (Nominal capacity-200MW).xlsx",
    "Wind farm site 3 (Nominal capacity-99MW).xlsx",
    "Wind farm site 4 (Nominal capacity-66MW).xlsx",
    "Wind farm site 5 (Nominal capacity-36MW).xlsx",
    "Wind farm site 6 (Nominal capacity-96MW).xlsx"
]

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Data import and model evaluation
for site_number in range(1, 7):
    file_name = xlsx_files[site_number - 1]
    data = pd.read_excel("../datasets/"+file_name)

    # Convert time column to datetime and correct invalid times
    data['Time(year-month-day h:m:s)'] = data['Time(year-month-day h:m:s)'].apply(lambda x: str(x).replace(' 24:', ' 00:'))
    data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'], format='%Y-%m-%d %H:%M:%S')

    # Set time column as index
    data.set_index('Time(year-month-day h:m:s)', inplace=True)

    # Strip leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Handle NaN values
    data.ffill(inplace=True)

    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

    # Prepare input/output
    X = data_scaled.iloc[:, :-1]  # All features except the last (target) column
    y = data_scaled.iloc[:, -1]   # Target column (Power output)

    # Create sequences
    time_steps = 256  # Match the context_length in xLSTM config
    X_seq, y_seq = create_sequences(X.values, y.values, time_steps)

    # Splitting data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create and train xLSTM model
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
    model = xLSTMLMModel(cfg)
    model = model.to('cuda')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to('cuda'), batch_y.to('cuda')
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs[:, -1], batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to('cuda')
            outputs = model(batch_X)
            predictions.extend(outputs[:, -1].cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(file_name)
    print(f" RMSE: {rmse}")
    print(f" MAE: {mae}")
    print(f" R2 Score: {r2}")

    # Save results to file
    results_file = "../outputs/wind_xlstm.txt"
    with open(results_file, "a") as file:
        file.write(f"Site {site_number}:\n")
        file.write(f"RMSE: {rmse}\n")
        file.write(f"MAE: {mae}\n")
        file.write(f"R2 Score: {r2}\n")
        file.write("\n")

    # # Plotting actual vs predicted
    # plt.figure(figsize=(10, 6), dpi=300)
    # plt.plot(actuals[:500], label='Actual Wind Power', linewidth=2)
    # plt.plot(predictions[:500], label='Predicted Wind Power', linewidth=2)
    # plt.title('Comparison of Actual and Predicted Wind Power (xLSTM)')
    # plt.xlabel('Time')
    # plt.ylabel('Wind Power (MW)')
    # plt.legend()
    # plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    # plt.savefig(f'Actual_vs_Predicted_Wind_Power_xLSTM_Site{site_number}.png')
    # plt.close()