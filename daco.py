# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import math, time
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import http.client
# Define the main function that takes a file path to the stock price data as input
def main(symbol,lookback):
    gru = []
    # Load stock data from CSV file and sort it by date
   


    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"}
    url = f'https://api.nasdaq.com/api/quote/{symbol}/historical?assetclass=stocks&fromdate=2014-04-06&limit=9999&todate=2024-04-06&api_key=MXhfTvQPCLT2nmSKDtQj'
    response = requests.get(url, headers=headers, timeout=5)
    a = response.json()["data"]["tradesTable"]
    print(a)
# Load the data into a pandas DataFrame
    data_stock = pd.DataFrame(response.json()["data"]["tradesTable"]["rows"])
    data_stock['close'] = data_stock['close'].str.replace('$', '')
    data_stock['date'] = pd.to_datetime(data_stock['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    print(data_stock)
    #data_stock = pd.read_csv(filepath)
    data_stock = data_stock.sort_values('date')
    
    # Define a function to split the data into sequences for training and testing
    def split_data(stock, lookback):
      
        data_raw = stock.to_numpy()
       
          # Convert DataFrame to numpy array
        data = []
        
        # Create all possible sequences of length equal to 'lookback'
        for index in range(len(data_raw) - lookback): 
            data.append(data_raw[index: index + lookback])
  
    
        data = np.array(data)
        test_set_size = int(np.round(0.2*data.shape[0]))  # Define test set size as 20% of the total
        train_set_size = data.shape[0] - test_set_size  # Remaining data will be training set
        
        # Split data into training and test sets for both features (x) and labels (y)
        #x_train2 = data[:train_set_size,:-1,:]
        #print(x_train2)
        adam = []
        index_raw = []
        for i in list(range(1,len(data_raw)+1)):
            index_raw.append([(i/200)])
        index_raw2 = index_raw.copy()
        for t in range(200):
            index_raw2.append([(t/200)])
        for index in range(len(data_raw) - lookback): 
            adam.append(index_raw[index: index + lookback])
        adam2=[]
        for index in range(len(index_raw2) - lookback): 
            adam2.append(index_raw2[index: index + lookback])
        
        adam = np.array(adam)
        adam2 = np.array(adam2)
        y_train = data[:train_set_size,-1,:]
        #print(type(y_train))
        x_train = adam[:train_set_size,:-1,:]
        #x_test = data[train_set_size:,:-1]
        x_test = adam[train_set_size:,:-1]
            
        #print(x_test)
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]

    # Select the 'Close' price column for prediction
    price_stock = data_stock[['close']]
    # Scale the 'Close' prices to the range (-1, 1) to normalize the input for the neural network
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print("hhhhhhhhhhhhhhhhh")
    print(price_stock['close'].values,"dacococo",type(price_stock['close'].values))
    print("bbbbbbbbb")
    price_stock['close'] = scaler.fit_transform(price_stock['close'].values.reshape(-1,1))
    print(price_stock['close'])
    # Define the dimensions and parameters for the GRU model
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 105
    
    # Define the GRU model class
    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # GRU layer
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        # Define the forward pass
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn) = self.gru(x, (h0.detach()))
            out = self.fc(out[:, -1, :])  # Take the last output for prediction
            return out

    # Set the lookback period to define how many previous time steps are used for prediction
    
    # Split the data into training and test sets
    x_train, y_train, x_test, y_test = split_data(price_stock, lookback)

    # Convert numpy arrays to torch tensors for training the model
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
    
    # Initialize the GRU model, loss function, and optimizer
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # Print training time
    training_time = time.time() - start_time    
    print("Training time: {}".format(training_time))
    
    # Visualize the training predictions and actual data
    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))
    #fig = plt.figure()
    # fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    # plt.subplot(1, 2, 1)
    # ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
    # ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    # ax.set_title('Stock price prediction', size = 14, fontweight='bold')
    # ax.set_xlabel("Days", size = 14)
    # ax.set_ylabel("Cost (USD)", size = 14)
    
    # # Visualize the training loss over epochs
    # plt.subplot(1, 2, 2)
    # ax = sns.lineplot(data=hist, color='royalblue')
    # ax.set_xlabel("Epoch", size = 14)
    # ax.set_ylabel("Loss", size = 14)
    # ax.set_title("Training Loss", size = 14, fontweight='bold')
    # fig.set_figheight(6)
    # fig.set_figwidth(16)


 

# make predictions
    # x_test = x_test.detach().numpy().tolist()
    # x_train_daco = x_train.detach().numpy().tolist()
    # print(type(x_test),type(x_train_daco))
    # for i in range(100):
    #     x_test.append(predictionssss.main(x_train_daco,x_test))
    # x_test.to_numpy()
    # x_test = torch.from_numpy(x_test).type(torch.Tensor)

    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    gru.append(trainScore)
    gru.append(testScore)
    gru.append(training_time)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price_stock)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price_stock)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price_stock)-1, :] = y_test_pred

    original = scaler.inverse_transform(price_stock['close'].values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)
   



    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                        mode='lines',
                        name='Train prediction')))
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                        mode='lines',
                        name='Actual Value')))
    fig.add_trace(go.Scatter(x=result.index, y=result[1],
                        mode='lines',
                        name='Test prediction'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template = 'plotly_dark'

    )



    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Apple Stock Prediction',
                                font=dict(family='Rockwell',
                                            size=26,
                                            color='white'),
                                showarrow=False))
    fig.update_layout(annotations=annotations)

    #fig.show()
    return(float(result[1].iloc[-2])/float(result[2].iloc[-500]))

# Call the main function with the path to the CSV file containing stock data
#main('AAPL',5)
import psycopg2

try:
    conn = psycopg2.connect(
        dbname='hk2024',
        user='hk2024_owner',
        password='abdcJTrNp68Q',
        host='ep-odd-truth-a27ydqp6-pooler.eu-central-1.aws.neon.tech'
    )
except psycopg2.Error as e:
    print("Error: Could not make connection to the Postgres database")
    print(e)
try:
    cur = conn.cursor()
except psycopg2.Error as e:
    print("Error: Could not get cursor to the Database")
    print(e)
try:
    cur.execute("SELECT symbol FROM nasdaq ORDER BY marekt_cap desc;")
except psycopg2.Error as e:
    print("Error: Could not execute query")
    print(e)
try:
    rows = cur.fetchall()
    for row in rows:
        try:
            print(row[0])
            data = main(row[0],20)
            cur.execute(f"UPDATE nasdaq SET ai_rating={data} WHERE symbol='{row[0]}'")
            conn.commit()
        except:
            pass
except psycopg2.Error as e:
    print("Error: Could not fetch data")
    print(e)