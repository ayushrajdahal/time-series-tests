import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Configuration for xLSTM
xlstm_cfg = """ 
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
cfg = OmegaConf.create(xlstm_cfg)
cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
xlstm_stack = xLSTMBlockStack(cfg).to("cuda")

# Data import and preprocessing
xlsx_files = [
    "Wind farm site 1 (Nominal capacity-99MW).xlsx",
    "Wind farm site 2 (Nominal capacity-200MW).xlsx",
    "Wind farm site 3 (Nominal capacity-99MW).xlsx",
    "Wind farm site 4 (Nominal capacity-66MW).xlsx",
    "Wind farm site 5 (Nominal capacity-36MW).xlsx",
    "Wind farm site 6 (Nominal capacity-96MW).xlsx"
]

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Evaluation
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
    X = data_scaled.iloc[:, :-1].values  # All features except the last (target) column
    y = data_scaled.iloc[:, -1].values   # Target column (Power output)

    # Splitting data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to("cuda")
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to("cuda")
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to("cuda")

    # Model training (placeholder, you need to adapt this part to the xLSTM API)
    xlstm_stack.train()
    optimizer = torch.optim.Adam(xlstm_stack.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        outputs = xlstm_stack(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Model evaluation
    xlstm_stack.eval()
    with torch.no_grad():
        predictions = xlstm_stack(X_test_tensor).cpu().numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(file_name)
    print(f" RMSE: {rmse}")
    print(f" MAE: {mae}")
    print(f" R2 Score: {r2}")

    # Save results to file
    results_file = "../outputs/wind_xlstm_v2.txt"
    with open(results_file, "a") as file:
        file.write(f"Site {site_number}:\n")
        file.write(f"RMSE: {rmse}\n")
        file.write(f"MAE: {mae}\n")
        file.write(f"R2 Score: {r2}\n")
        file.write("\n")

    # # Plotting actual vs predicted
    # plt.figure(figsize=(10, 6), dpi=300)
    # plt.plot(y_test[:500], label='Actual Wind Power', linewidth=2)
    # plt.plot(predictions[:500], label='Predicted Wind Power', linewidth=2)
    # plt.title('Comparison of Actual and Predicted Wind Power')
    # plt.xlabel('Time')
    # plt.ylabel('Wind Power (MW)')
    # plt.legend()
    # plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    # plt.savefig('Actual_vs_Predicted_Wind_Power_XLSTM.png')
    # plt.show()

    # # Feature importance analysis (adapt according to xLSTM, if applicable)
    # feature_importance = model.feature_importances_
    # feature_names = X.columns
    #
    # plt.figure(figsize=(10, 6), dpi=300)
    # plt.bar(feature_names, feature_importance)
    # plt.ylabel('Feature Importance', fontsize=12, fontweight='bold')
    # plt.xticks(rotation=45, fontsize=12, fontweight='bold', ha='right')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.tight_layout()
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # plt.savefig('Feature_Importance_XLSTM.png')
    # plt.show()
