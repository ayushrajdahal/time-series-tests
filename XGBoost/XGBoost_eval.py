import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import re

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

    # Extract nominal capacity using regular expression pattern from the file name
    match = re.search(r"Nominal capacity-(\d+)MW", file_name)
    if match:
        nominal_capacity = int(match.group(1))
        print(f"Nominal capacity: {nominal_capacity} MW")
    else:
        print("Nominal capacity not found")


    data = pd.read_excel("../datasets/"+file_name)

    # Convert time column to datetime and correct invalid times
    data['Time(year-month-day h:m:s)'] = data['Time(year-month-day h:m:s)'].apply(lambda x: str(x).replace(' 24:', ' 00:'))
    data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'], format='%Y-%m-%d %H:%M:%S')


    # Extract month-day data and convert to integer day value
    data['DayOfYear'] = data['Time(year-month-day h:m:s)'].dt.dayofyear

    # Convert time to fractional hour data
    data['FractionalHour'] = data['Time(year-month-day h:m:s)'].dt.hour + data['Time(year-month-day h:m:s)'].dt.minute / 60.0

    # Set time column as index
    data.set_index('Time(year-month-day h:m:s)', inplace=True)

    # Strip leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Handle NaN values
    data.ffill(inplace=True)

    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.iloc[:, :-1])
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1], index=data.index)

    # Add the target column to the scaled data w/o normalizing

    data_scaled['Power (MW)'] = data['Power (MW)']

    # Prepare input/output
    X = data_scaled.iloc[:, :-1]  # All features except the last (target) column
    y = data_scaled.iloc[:, -1]   # Target column (Power output)
    y = y / nominal_capacity

    # Splitting data into training, validation, and testing sets (70-15-15 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create and train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(file_name)
    print(f" RMSE: {rmse}")
    print(f" MAE: {mae}")
    print(f" R2 Score: {r2}")

    # Save results to file
    results_file = "../outputs/wind_xgboost.txt"
    with open(results_file, "a") as file:
        file.write(f"{file_name.split(".xlsx")[0]}:\n")
        file.write(f" RMSE: {rmse}\n")
        file.write(f" MAE: {mae}\n")
        file.write(f" R2 Score: {r2}\n")
        file.write("\n")

    # # Plotting actual vs predicted
    # plt.figure(figsize=(10, 6), dpi=300)
    # plt.plot(y_test[:500].values, label='Actual Wind Power', linewidth=2)
    # plt.plot(predictions[:500], label='Predicted Wind Power', linewidth=2)
    # plt.title('Comparison of Actual and Predicted Wind Power')
    # plt.xlabel('Time')
    # plt.ylabel('Wind Power (MW)')
    # plt.legend()
    # plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    # plt.savefig('Actual_vs_Predicted_Wind_Power_XGBoost.png')
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
    # plt.savefig('Feature_Importance_XGBoost.png')
    # plt.show()