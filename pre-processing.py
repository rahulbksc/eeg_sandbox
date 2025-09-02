import mne
import numpy as np

# 1. Load raw EEG data (update filepath and file format as needed)
raw = mne.io.read_raw_fif('your_file_raw.fif', preload=True)  
# Alternative: read_raw_edf('your_file.edf'), etc.

# 2. Apply Bandpass Filter (e.g., 1-50 Hz for general EEG analysis)
raw.filter(l_freq=1.0, h_freq=50.0)

# 3. Notch Filter to remove powerline noise (e.g., 50 or 60 Hz)
raw.notch_filter(freqs=np.arange(50, 251, 50))  # Use 60 for US powerline

# 4. Resample the signal (e.g., downsample to 256 Hz)
raw.resample(256)

# 5. Detect and mark bad channels (manual/automatic; here, mark by name)
raw.info['bads'].extend(['EEG 001', 'EEG 002'])  # Replace with your bad channel names

# 6. Interpolate bad channels
raw.interpolate_bads()

# 7.
''' 
    Detect and remove ocular/muscle artifacts via ICA to separate the EEG signals into independent components.
    which often correspond to distinct physiological or noise sources (e.g., eye blinks, muscle activity, heartbeats).
'''
ica = mne.preprocessing.ICA(n_components=20, random_state=42)

#Decompose the signal into statistically independent sources.
ica.fit(raw)

#Identify and remove components corresponding to common signal artifacts (e.g., eye blinks, muscle movements).
ica.detect_artifacts(raw)

'''
Transform the Data: The function projects the EEG signal into the ICA (independent components) space.
Zeroes Artifact Components: Any ICA components marked for exclusion (commonly those identified as artifacts) 
are set to zero within the ICA space, effectively removing their influence from the data.
Reconstructs Cleaned Data: The data is then projected back into the original signal space, 
but now without the contributions from the excluded artifact components.
'''
raw = ica.apply(raw)

# 8. Set EEG average reference
'''
Average reference - Instead of referencing to a single electrode, each EEG channel is re-referenced to the average potential 
across all available channels. This is a common strategy for reducing bias and improving data comparability in EEG studies.
projection=True: Rather than immediately modifying the data, this adds a projection operator to the data object. 
This means the average reference is only applied when the signal is accessed, keeping the raw data untouched, 
which allows flexible re-referencing later if needed.

'''
raw.set_eeg_reference('average', projection=True)

# 9. Save the preprocessed data
raw.save('preprocessed_raw.fif', overwrite=True)