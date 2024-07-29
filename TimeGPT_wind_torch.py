import torch
from nixtla import NixtlaClient
import pandas as pd


API_KEY = 'nixtla-tok-7FJsZBnuXE04TuDFi0UrUmeoV03oHl9FJbe0h7cxDMiyIFuch5RvIcoKGQki5erTO05vHh2TDtFJQpJ1'
nixtla_client = NixtlaClient(api_key=API_KEY)

df = pd.read_xlsx('datasets/Wind farm site 1 (Nominal capacity-99MW).xlsx')