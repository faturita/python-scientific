"""
==================
Kalman Filter
==================

A Kalman filter is a recursive Bayesian estimator for linear Gaussian
state-space models.  It alternates between two steps:

  Predict  — propagate the state estimate forward using the process model.
  Update   — correct the prediction with the new measurement.

State-space model used here (1-D, constant-velocity random walk):

  x_k  = F * x_{k-1} + w_k       w_k  ~ N(0, Q)   process noise
  z_k  = H * x_k     + v_k       v_k  ~ N(0, R)   measurement noise

With F = H = 1 this reduces to a scalar random-walk model, which is the
simplest sensible model for a slowly-varying EEG baseline.

Tuning knobs
------------
Q   — process noise variance.  Larger Q lets the filter track fast changes
      but admits more noise.  Smaller Q gives a smoother, lazier estimate.
R   — measurement noise variance.  Should reflect sensor noise power.
      If R >> Q the filter trusts the model; if R << Q it trusts measurements.

The ratio Q/R is what actually matters; you can normalise one to 1 and
sweep the other.

This script is a stepping-stone toward a Particle Filter: the Kalman filter
is the analytical solution to the same problem when the model and noise are
both Gaussian.  The particle filter drops that restriction.

References
----------
* Welch & Bishop, "An Introduction to the Kalman Filter", UNC TR 95-041.
* Thrun, Burgard & Fox, "Probabilistic Robotics", Ch. 3.
"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Data ────────────────────────────────────────────────────────────────────

signals = pd.read_csv(
    'data/blinking.dat',
    delimiter=' ',
    names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking']
)

data = signals.values
eeg  = data[:, 2].astype(float)      # raw single-channel EEG
N    = len(eeg)

print(f'Loaded {N} samples.  EEG range: [{eeg.min():.1f}, {eeg.max():.1f}]')

# ── Kalman filter ────────────────────────────────────────────────────────────

def kalman_filter(z, Q=1e-3, R=1.0, x0=None, P0=1.0):
    """
    Scalar 1-D Kalman filter (F = H = 1, random-walk process model).

    Parameters
    ----------
    z   : array-like, shape (N,)
          Noisy measurements (raw EEG samples).
    Q   : float
          Process noise variance.  Governs how quickly the true state can change.
    R   : float
          Measurement noise variance.  Governs how much we trust each sample.
    x0  : float, optional
          Initial state estimate (defaults to first measurement).
    P0  : float
          Initial error covariance (uncertainty in x0).

    Returns
    -------
    x_hat : ndarray, shape (N,)
            Filtered state estimates.
    K_log : ndarray, shape (N,)
            Kalman gain at each step (diagnostic).
    P_log : ndarray, shape (N,)
            Posterior error covariance at each step (diagnostic).
    """
    z = np.asarray(z, dtype=float)
    n = len(z)

    x_hat = np.empty(n)   # filtered estimates
    K_log = np.empty(n)   # Kalman gains
    P_log = np.empty(n)   # error covariances

    # Initialise
    x = z[0] if x0 is None else float(x0)
    P = float(P0)

    for k in range(n):
        # ── Predict ──────────────────────────────────────────────────────────
        # With F = 1: x_pred = x,  P_pred = P + Q
        x_pred = x
        P_pred = P + Q

        # ── Update ───────────────────────────────────────────────────────────
        # Innovation (residual)
        y = z[k] - x_pred                   # measurement residual

        # Innovation covariance  S = H*P_pred*H^T + R  (H=1)
        S = P_pred + R

        # Kalman gain  K = P_pred * H^T * S^{-1}
        K = P_pred / S

        # State update
        x = x_pred + K * y

        # Covariance update  (Joseph form for numerical stability)
        P = (1.0 - K) * P_pred

        x_hat[k] = x
        K_log[k] = K
        P_log[k] = P

    return x_hat, K_log, P_log


# ── Run with three Q/R settings to show the trade-off ───────────────────────

configs = [
    dict(Q=1e-5, R=5.0,  label='Smooth  (Q=1e-5, R=5)',   color='green'),
    dict(Q=1e-2, R=1.0,  label='Balanced (Q=1e-2, R=1)',   color='blue'),
    dict(Q=1.0,  R=0.1,  label='Reactive (Q=1,    R=0.1)', color='orange'),
]

results = []
for cfg in configs:
    x_hat, K_log, P_log = kalman_filter(eeg, Q=cfg['Q'], R=cfg['R'])
    results.append((x_hat, K_log, P_log, cfg))
    print(f"  {cfg['label']:40s}  mean K={K_log.mean():.4f}  final P={P_log[-1]:.6f}")

# ── Plot 1: raw vs. filtered signals ─────────────────────────────────────────

fig, axes = plt.subplots(len(configs) + 1, 1, figsize=(14, 10), sharex=True)

axes[0].plot(eeg, 'r', linewidth=0.6, label='Raw EEG')
axes[0].set_ylabel('EEG (raw)')
axes[0].set_title('Raw EEG signal')
axes[0].set_ylim([-2000, 2000])
axes[0].legend(loc='upper right')

for ax, (x_hat, _, _, cfg) in zip(axes[1:], results):
    ax.plot(eeg,   'r',          linewidth=0.4, alpha=0.4, label='Raw')
    ax.plot(x_hat, cfg['color'], linewidth=1.2, label=f"Kalman — {cfg['label']}")
    ax.set_ylabel('EEG (filtered)')
    ax.set_ylim([-2000, 2000])
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Sample index')
fig.suptitle('Kalman Filter — EEG denoising (Q/R trade-off)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/kalman_eeg.png', dpi=150)
plt.show()

# ── Plot 2: Kalman gain and error covariance convergence ─────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

for x_hat, K_log, P_log, cfg in results:
    ax1.plot(K_log, label=cfg['label'], color=cfg['color'])
    ax2.plot(P_log, label=cfg['label'], color=cfg['color'])

ax1.set_ylabel('Kalman gain K')
ax1.set_title('Kalman gain convergence')
ax1.legend()

ax2.set_ylabel('Error covariance P')
ax2.set_title('Posterior error covariance')
ax2.set_xlabel('Sample index')
ax2.legend()

fig.suptitle('Kalman Filter diagnostics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/kalman_diagnostics.png', dpi=150)
plt.show()

print('\nDone.  Plots saved to images/kalman_eeg.png and images/kalman_diagnostics.png')
