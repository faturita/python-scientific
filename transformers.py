"""
El objetivo de este experimento es procesar y preparar datos EEG para entrenar un modelo Transformer
que clasifique diferentes estados mentales o actividades basadas en series temporales de señales EEG.
Primero, se filtra y organiza los datos EEG, los etiqueta y los convierte en secuencias temporales.
Luego, se crea un modelo Transformer donde se entrena utilizando estas secuencias,
para clasificar los estados o actividades registradas en los datos EEG.

El largo de los registros es entre 10 y 11 minutos
Fs = 512

|---- BASELINE --------|
|---- PESTANEO ------|
|---- INHALAR ------- |
|---- EXHALAR ----|
|---- GOLPES (ESTIMULOS) --------|
|---- MENTAL IMAGERY ------|
|---- CERRADOS --------|

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Inhalar: Cambios en frecuencias bajas, donde se enfatice el movimiento del diafragma para inhalar.
* Exhalar: Idem pero enfatizando el movimiento del diafragma para exhalar.
* Golpes: Deberían aparecer algún evento con cierto pico en la señal.  Pueden intentar detectar estos picos y ver si hay algo.
* Mental Imagery: Pueden aparecer frecuencias altas de 20-50 Hz.  Aumentos en la potencia de estas bandas entre este bloque y el baseline.
* Cerrados: Debería aparecer un aumento en la frecuencia particular de 10 Hz en relación al baseline.

Primera parte: PREPARAR EL DATASET

En esta primera parte, se carga y procesa datos EEG de varios archivos,
aplicando filtros pasabanda para aislar ciertas frecuencias relevantes en la señal EEG. 
Luego, recorta y divide los datos en particiones iguales para cada condición. 
Cada partición se transpone, convirtiendo las filas de datos EEG en columnas.

Finalmente, combina todas las particiones en un solo DataFrame, 
agregando una columna que etiqueta cada conjunto de datos según la condición experimental correspondiente. 
Este DataFrame final, estaría listo para ser utilizado para la segunda parte
que es el desarrollo del modelo de transformers

Segunda parte: TRANSFORMERS

En esta parte se entrena un modelo de Transformer para clasificar
series temporales basadas en datos EEG.
Primero se organiza los datos en secuencias de longitud fija (30)
para formar un conjunto de datos que se puede alimentar al modelo.
Luego, se utiliza un DataLoader para manejar el muestreo y
la preparación de lotes de datos durante el entrenamiento.

Durante el entrenamiento, el modelo calcula
la pérdida utilizando cross-entrpy y
ajusta los pesos usando Adam.

IMPORTANTE: instalar PyTorch para el desarrollo de transformers

pip install torch
conda install torch

"""

"""
Primera parte: PREPARAR EL DATASET
"""

import pandas as pd
from scipy.signal import butter, lfilter

column_names = ['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking']
baseline = pd.read_csv('data/072024_muestras_EEG/baseline.dat', delimiter=' ', names=column_names)
exhalar = pd.read_csv('data/072024_muestras_EEG/exhalar.dat', delimiter=' ', names=column_names)
golpes1 = pd.read_csv('data/072024_muestras_EEG/golpes1.dat', delimiter=' ', names=column_names)
golpes2 = pd.read_csv('data/072024_muestras_EEG/golpes2.dat', delimiter=' ', names=column_names)
cerrados = pd.read_csv('data/072024_muestras_EEG/cerrados.dat', delimiter=' ', names=column_names)
mentalimagery = pd.read_csv('data/072024_muestras_EEG/mentalimagery.dat', delimiter=' ', names=column_names)
pestaneos = pd.read_csv('data/072024_muestras_EEG/pestaneos.dat', delimiter=' ', names=column_names)
inhalar = pd.read_csv('data/072024_muestras_EEG/inhalar.dat', delimiter=' ', names=column_names)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

mentalimagery['eeg'] = butter_bandpass_filter(mentalimagery['eeg'], 10, 50, 512, 5)
cerrados['eeg'] = butter_bandpass_filter(cerrados['eeg'], 5, 15, 512, 5)
inhalar['eeg'] = butter_bandpass_filter(inhalar['eeg'], 1, 15, 512, 5)
exhalar['eeg'] = butter_bandpass_filter(exhalar['eeg'], 1, 15, 512, 5)

partition_number = 100

#dataframes = [baseline, exhalar, golpes1, golpes2, cerrados, mentalimagery, pestaneos, inhalar]
dataframes = [baseline, exhalar, cerrados, mentalimagery, pestaneos, inhalar]
min_length = min(df.shape[0] for df in dataframes)
dataframes_trimmed = [df.iloc[:min_length] for df in dataframes]

def divide_into(df):
    rows_per_df = len(df) // partition_number
    return [df.iloc[i*rows_per_df: (i+1)*rows_per_df].reset_index(drop=True) for i in range(partition_number)]

divided_dataframes = {}
for i, df in enumerate(dataframes_trimmed):
    divided_dataframes[f'dataframe_{i+1}'] = divide_into(df)

def transpose_eeg_dataframe(df):
    transposed_df = df['eeg'].to_frame().T
    transposed_df.columns = [f'row_{i}' for i in df.index]
    return transposed_df

targets = {
    'dataframe_1': 'baseline',
    'dataframe_2': 'exhalar',
    #'dataframe_3': 'golpes1',
    #'dataframe_4': 'golpes2',
    'dataframe_3': 'cerrados',
    'dataframe_4': 'mentalimagery',
    'dataframe_5': 'pestaneos',
    'dataframe_6': 'inhalar'
}

all_combined_dataframes = []
for key, target in targets.items():
    dataframe_list = divided_dataframes[key]
    transposed_dataframes = []
    for df in dataframe_list:
        transposed_df = transpose_eeg_dataframe(df)
        transposed_dataframes.append(transposed_df)
    combined_dataframe = pd.concat(transposed_dataframes, ignore_index=True)
    combined_dataframe['target'] = target
    all_combined_dataframes.append(combined_dataframe)

final_combined_dataframe = pd.concat(all_combined_dataframes, ignore_index=True)

print("Data frame final: ")
print(final_combined_dataframe)

"""
Segunda parte: TRANSFORMERS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

data = final_combined_dataframe

label_encoder = LabelEncoder()
data['target'] = label_encoder.fit_transform(data['target'])

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        self.features = data.columns[:-1]  # Exclude the target column

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[self.features].iloc[idx:idx+self.sequence_length].values
        y = self.data['target'].iloc[idx+self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

sequence_length = 30
dataset = TimeSeriesDataset(data, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size * sequence_length, num_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transpose for transformer input
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # Flatten
        x = self.fc(x)
        return x

input_size = data.shape[1] - 1
num_layers = 2
num_heads = 4
hidden_dim = 128
num_classes = len(label_encoder.classes_)

model = TransformerModel(input_size, num_layers, num_heads, hidden_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}')

print("Entrenamiento completado.")
