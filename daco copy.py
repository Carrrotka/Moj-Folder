# Step 0: Install necessary packages (run these commands in your terminal)
# pip install torch torchvision
# pip install matplotlib

# Step 1: Prepare Your Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('AAPL-Data2.csv')
df['Date'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)  # Convert dates

# Normalize values
scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_set, test_set = df[:train_size], df[train_size:]

# Convert to PyTorch tensors
print(train_set)
train_features = torch.tensor(train_set['Date'].values.astype(float)).float()
train_targets = torch.tensor(train_set['Close'].values.astype(float)).float()
test_features = torch.tensor(test_set['Date'].values.astype(float)).float()
test_targets = torch.tensor(test_set['Close'].values.astype(float)).float()

# Create dataset and dataloader for training
train_dataset = TensorDataset(train_features, train_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 2: Define Your Model
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1)  # Adjust sizes as needed

    def forward(self, x):
        return self.linear(x)

model = SimpleNN()

# Step 3: Train Your Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):  # Number of epochs
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))  # Adjust input shape as needed
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 4: Evaluate and Predict
# Make predictions
with torch.no_grad():  # We do not need to track gradients here
    model.eval()  # Set the model to evaluation mode
    predictions = model(test_features.unsqueeze(1)).squeeze(1)  # Adjust input shape as needed

# Inverse transform predictions and actual values if you've normalized your data
predictions_inverse = scaler.inverse_transform(predictions.numpy().reshape(-1, 1))
actual_inverse = scaler.inverse_transform(test_targets.detach().numpy().reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual_inverse, label='Actual Values')
#plt.plot(predictions_inverse, label='Predicted Values', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()