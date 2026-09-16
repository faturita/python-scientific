import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Step (i) – synthetic signal: oscillatory → transient → non-oscillatory
fs = 500
t = np.linspace(0, 10, fs * 10)

osc1 = np.sin(2 * np.pi * 5 * t) * (t < 3)
transient = np.exp(-((t - 5)**2) / (2 * 0.1**2))
decay = np.exp(-0.5 * (t - 6)) * (t > 6)
signal = osc1 + transient + decay

signal = np.sin(2 * np.pi * 5 * t)  

# Step (ii) – analytic signal and demodulation
analytic = hilbert(signal)
amplitude = np.abs(analytic)
phase = np.unwrap(np.angle(analytic))

carrier_phase = np.copy(phase)
carrier_phase[t >= 3] = phase[np.where(t == t[t < 3][-1])[0][0]]  # freeze phase
demodulated = analytic * np.exp(-1j * carrier_phase)

# Step (iii) – compute local “oscillatoriness index” via SVD fit error
win_size = fs  # 1-second sliding window
step = fs // 10
indices = np.arange(0, len(t) - win_size, step)
osc_index = []

for i in indices:
    w = amplitude[i:i + win_size]
    if len(w) < 2:
        continue
    A = w.reshape(-1, 1)
    U, S, Vt = np.linalg.svd(A - np.mean(A), full_matrices=False)
    recon = np.mean(A) + S[0] * U[:, 0] * Vt[0, 0]
    residual = np.linalg.norm(A - recon) / np.linalg.norm(A)
    osc_index.append(1 - residual)  # 1→perfect oscillation, 0→non-oscillatory

t_index = t[indices + win_size // 2]
osc_index = np.array(osc_index)

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")

plt.subplot(4, 1, 2)
plt.plot(t, amplitude)
plt.title("Amplitude Envelope (|analytic|)")

plt.subplot(4, 1, 3)
plt.plot(t, np.abs(demodulated))
plt.title("Demodulated Magnitude (slowly varying)")

plt.subplot(4, 1, 4)
plt.plot(t_index, osc_index, label="Oscillatoriness Index", color='purple')
plt.ylim(0, 1.1)
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.show()