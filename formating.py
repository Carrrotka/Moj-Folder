import pandas as pd
from datetime import datetime

# Load your CSV file
df = pd.read_csv('AAPL-Data.csv')

# Assuming 'date_column' is the name of the column containing the dates
# Convert the dates to the desired format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

# Save the modified DataFrame back to a CSV
df.to_csv('AAPL-Data2.csv', index=False)