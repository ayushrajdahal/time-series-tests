import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import TSLANet model
from TSLANet_Forecasting import TSLANet

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

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

# Data import
for site_number, file_name in enumerate(xlsx_files, 1):
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

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize TSLANet model
    model = TSLANet(input_dim=X_train.shape[1], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(20):  # Number of epochs
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().numpy())

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(file_name)
    print(f" RMSE: {rmse}")
    print(f" MAE: {mae}")
    print(f" R2 Score: {r2}")

    # Save results to file
    results_file = "../outputs/wind_tsla_net.txt"
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
    # plt.savefig('Actual_vs_Predicted_Wind_Power_TSLANet.png')
    # plt.show()

    # # Feature importance analysis
    # feature_importance = model.feature_importances_
    # feature_names = X.columns

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
    # plt.savefig('Feature_Importance_TSLANet.png')
    # plt.show()