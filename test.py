
import csv
import requests
import json
import pandas as pd

# Replace <API_KEY> with your actual API key for the JSONPlaceholder API


# Make a GET request to the API
response = requests.get('https://api.nasdaq.com/api/quote/GOOGL/historical?assetclass=stocks&fromdate=2014-04-06&limit=9999&todate=2024-04-06&api_key=MXhfTvQPCLT2nmSKDtQj')

# Load the data into a pandas DataFrame
data = pd.read_json(response.content)

# Display the DataFrame
print(data)