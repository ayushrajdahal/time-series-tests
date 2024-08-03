import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from XLSTM_custom import XLSTM_Attention_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create sequences for time series data
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :-1])  # All features except the last (target) column
        y.append(data[i, -1])  # Target column (Power output)
    return np.array(X), np.array(y)
 
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
    data = pd.read_excel('../datasets/'+file_name)  # Adjust file path as needed
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
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = XLSTM_Attention_Model(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
            true_values.extend(labels.cpu().numpy())
    
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    
    # Save results to file
    results_file = "../outputs/results_n72_XLSTM_Attention.txt"
    with open(results_file, "a") as file:
        file.write(f"{file_name.split(".xlsx")[0]}:\n")
        file.write(f" RMSE: {rmse}\n")
        file.write(f" MAE: {mae}\n")
        file.write(f" R2 Score: {r2}\n")
        file.write("\n")
    
    # Plot predictions vs true values
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.title(f'Solar Power Forecasting - Site {site_number}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Power Output')
    plt.legend()
    plt.savefig(f'../outputs/solar_forecast_site_{site_number}.png')
    # plt.show()