import mne
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from antropy import spectral_entropy

# 1. Load preprocessed data
raw = mne.io.read_raw_fif('preprocessed_raw.fif', preload=True)

# 2. Define epochs (as required, here around events)
'''
automatically detects and extracts event markers (triggers) from the raw EEG recording, returning them as an array. 
These events typically represent stimuli presentations, participant responses, or experimental conditions that 
were logged during data acquisition.
Returns a 2D NumPy array, shape (n_events, 3), where each row corresponds to a single event:
Column 0: Sample index at which the event occurred
Column 1: Value of the stim channel before the event
Column 2: Value of the stim channel after the event (usually the event code/ID)
'''
events = mne.find_events(raw)
event_id = {'stimulus': 1}  # adjust to your event marker

'''
creates an Epochs object, segmenting the continuous EEG data into short, 
time-locked slices (epochs) around specified events. It generates a container of shape (n_epochs, n_channels, n_times), 
where each “epoch” contains EEG data from a 2-second window after each event of the specified type. 
This is the standard input for most feature extraction and classification workflows
'''
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=2, baseline=None, preload=True)

'''
These dictionary entries assign the standard frequency ranges (in Hz) 
that correspond to well-known EEG brain wave bands.
'''
# 3. Define frequency bands (Hz)
bands = {'delta': (1, 4),
         'theta': (4, 8),
         'alpha': (8, 13),
         'beta' : (13, 30),
         'gamma': (30, 45)}

# 4. Feature extraction for each epoch and channel
'''
Extracts band power, statistical, and spectral entropy features from every channel and epoch, 5
assembling them into vectors for subsequent classification or analysis.

For Each Epoch and Channel:
The code iterates over each epoch (a 2-second slice of EEG data) and then over each channel within an epoch.
Feature Calculation:
For each channel in each epoch:
    Band Power:             Uses Welch's method to estimate the signal's 
                            power spectral density (PSD), then sums the power within 
                            each pre-defined frequency band (delta, theta, alpha, beta, gamma).

    Statistical Features:   Computes mean, standard deviation, 
                            kurtosis, and skewness of the channel's data.

    Spectral Entropy:       Calculates the entropy of the channel's power spectrum, 
                            quantifying how “random” or “complex” the frequency content is.
'''
features = []

for epoch in epochs.get_data():  # shape: (n_channels, n_times)
    channel_features = []
    for ch in epoch:
        # Band Power via Welch's method
        ch_bandpower = []
        freqs, psd = welch(ch, fs=raw.info['sfreq'], nperseg=256)
        for band, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            bp = np.sum(psd[idx_band])
            ch_bandpower.append(bp)
        # Statistical features
        ch_mean = np.mean(ch)
        ch_std = np.std(ch)
        ch_kurtosis = kurtosis(ch)
        ch_skewness = skew(ch)
        # Spectral entropy
        ch_entropy = spectral_entropy(ch, sf=raw.info['sfreq'], method='welch', normalize=True)
        # Aggregate
        ch_feats = ch_bandpower + [ch_mean, ch_std, ch_kurtosis, ch_skewness, ch_entropy]
        channel_features.append(ch_feats)
    features.append(np.hstack(channel_features))  # Optional: flatten for ML

features = np.array(features)  # Shape: epochs x (channels*features)

#print('Features shape:', features.shape)
