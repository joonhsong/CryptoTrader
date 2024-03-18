import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.model import Informer
from utils.timefeatures import time_features
import pyupbit

class iDataset(Dataset):
    def __init__(self, df, enc_seq_len, dec_seq_len, time_column, value_column):
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.df = df
        self.value_data = torch.tensor(df[value_column].values, dtype=torch.float32)

        self.time_data = pd.to_datetime(df[time_column])
        self.year = torch.tensor(self.time_data.dt.year.to_numpy(), dtype=torch.float32)
        self.month = torch.tensor(self.time_data.dt.month.to_numpy(), dtype=torch.float32)
        self.day = torch.tensor(self.time_data.dt.day.to_numpy(), dtype=torch.float32)
        self.hour = torch.tensor(self.time_data.dt.hour.to_numpy(), dtype=torch.float32)
        self.minute = torch.tensor(self.time_data.dt.minute.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.df) - self.enc_seq_len - self.dec_seq_len + 1 

    def __getitem__(self, idx):
        enc_idx = idx + self.enc_seq_len
        dec_idx = enc_idx + self.dec_seq_len

        enc_value_seq = self.value_data[idx:enc_idx]
        enc_time_seq = torch.stack((self.year[idx:enc_idx], 
                                    self.month[idx:enc_idx], 
                                    self.day[idx:enc_idx], 
                                    self.hour[idx:enc_idx], 
                                    self.minute[idx:enc_idx]), dim=-1)

        dec_value_seq = self.value_data[enc_idx:dec_idx]
        dec_time_seq = torch.stack((self.year[enc_idx:dec_idx], 
                                    self.month[enc_idx:dec_idx], 
                                    self.day[enc_idx:dec_idx], 
                                    self.hour[enc_idx:dec_idx], 
                                    self.minute[enc_idx:dec_idx]), dim=-1)
        
        print(f'enc_time_seq shape: {enc_time_seq.shape}')
        print(f'dec_time_seq shape: {dec_time_seq.shape}')

        return (enc_time_seq, enc_value_seq), (dec_time_seq, dec_value_seq)




def predict_future(model, input_data, input_mark, future_steps, scaler):
    model.eval()
    with torch.no_grad():
        input_data = input_data.to(model.device)
        input_mark = input_mark.to(model.device)

        input_data = input_data.unsqueeze(0)
        input_mark = input_mark.unsqueeze(0)

        pred_values = model(input_data, input_mark, input_data, input_mark)

        pred_values = pred_values.squeeze(0)

        pred_values = scaler.inverse_transform(pred_values.cpu().numpy())

        return pred_values
    
df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=180)
mms = MinMaxScaler()
temp = df['close'].reset_index().to_dict('list')
temp['close'] = mms.fit_transform(df['close'].to_numpy().reshape(-1,1)).reshape(-1)
data = pd.DataFrame(temp, columns=['index', 'close'])
data.columns = ['time', 'close']
data['time'] = pd.to_datetime(data['time'])
seq_len = 100  
label_len = 10  
pred_len = 10  
time_column = 'time'
value_column = 'close'
print(data)

data_train = data.iloc[:-40].copy()
data_val = data.iloc[-(seq_len + pred_len):].copy()

dataset_train =  iDataset(data_train, seq_len, label_len, time_column, value_column)
dataset_val = iDataset(data_val, seq_len, label_len, time_column, value_column)

# Notice the change in dec_in value from 1 to 5
model = Informer(enc_in=5,  
                 dec_in=5,  
                 c_out=1,   
                 seq_len=96,    
                 label_len=48,  
                 out_len=24,    
                 factor=5,  
                 d_model=512,   
                 n_heads=8, 
                 e_layers=3,    
                 d_layers=2,    
                 d_ff=512,  
                 dropout=0.05,  
                 attn='prob',   
                 embed='fixed',  
                 freq='h',   
                 activation='gelu',  
                 output_attention=False, 
                 distil=True,   
                 device=torch.device('cpu') 
                 )

device = torch.device('cpu')
model.to(device)

batch_size = 32
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

loss_func = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, ((enc_time_seq, enc_value_seq), (dec_time_seq, dec_value_seq)) in enumerate(train_dataloader):
        enc_time_seq = enc_time_seq.to(device)
        dec_time_seq = dec_time_seq.to(device)
        enc_value_seq = enc_value_seq.unsqueeze(-1).to(device).float()
        dec_value_seq = dec_value_seq.unsqueeze(-1).to(device).float()

        optimizer.zero_grad()
        print(enc_time_seq.shape, enc_value_seq.shape, dec_time_seq.shape, dec_value_seq.shape)

        outputs = model(enc_time_seq, enc_value_seq, dec_time_seq, dec_value_seq)

        loss = loss_func(outputs, dec_value_seq)

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch}, Step {i}, Loss: {loss.item()}')

model.eval()
val_loss = 0.0
with torch.no_grad():
    for i, ((enc_time_seq, enc_value_seq), (dec_time_seq, dec_value_seq)) in enumerate(val_dataloader):
        enc_time_seq = enc_time_seq.to(model.device)
        enc_value_seq = enc_value_seq.unsqueeze(-1).to(model.device)
        dec_time_seq = dec_time_seq.to(model.device)
        dec_value_seq = dec_value_seq.unsqueeze(-1).to(model.device)
        
        outputs = model(enc_time_seq, enc_value_seq, dec_time_seq, dec_value_seq)
        
        loss = loss_func(outputs, dec_value_seq)

        val_loss += loss.item()
    
val_loss /= len(val_dataloader)
print(f'Validation Loss: {val_loss}')

input_data, input_mark = dataset_val[-1]

future_steps = 10
pred_values = predict_future(model, input_data, input_mark, future_steps, mms)

print(pred_values)
