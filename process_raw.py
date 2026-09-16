import pandas as pd
import numpy as np
import mne

data1 = pd.read_csv('01.csv', delimiter=',', names=['id','eeg','timestamp'])
data2 = pd.read_csv('02.csv', delimiter=',', names=['id','eeg','timestamp'])

signals1 = data1['eeg'].values
signals2 = data2['eeg'].values

# Convert the signals to numpy arrays
signals1 = np.reshape(signals1, (1,len(signals1)))
signals2 = np.reshape(signals2, (1,len(signals2)))

# Create a new MNE Info object
info = mne.create_info(ch_names=['Fz'], sfreq=250, ch_types=['eeg'])

# Create MNE RawArray objects
raw1 = mne.io.RawArray(signals1, info)
raw2 = mne.io.RawArray(signals2, info)

raw1.filter(1,30)
raw2.filter(1,30)

raw1.plot_psd()
raw2.plot_psd()    


raw1.plot()
raw2.plot(block=True)


