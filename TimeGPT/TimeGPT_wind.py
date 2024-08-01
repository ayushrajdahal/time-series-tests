import os
from nixtla import NixtlaClient
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nixtla.utils

# validate api key

API_KEY = os.getenv('NIXTLA_API_KEY')
print(f'API_KEY: {API_KEY}')
nixtla_client = NixtlaClient(api_key=API_KEY)
valid_api_key = nixtla_client.validate_api_key()
print(f'{valid_api_key=}')

