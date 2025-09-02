# !pip install numpy scipy scikit-learn matplotlib tensorflow mne

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --------- Data Preparation ---------

# Dummy data parameters
n_epochs = 1000
epoch_length = 3000  # 30 seconds at 100Hz
n_classes = 5

# Synthetic EEG epochs and labels (replace with real data)
eeg_epochs = np.random.randn(n_epochs, epoch_length)
labels = np.random.randint(0, n_classes, n_epochs)

# ----------- FEATURE EXTRACTION FOR ML ------------
# Extracts classic EEG features (Band power)

def compute_bandpower(epoch, sf=100, band=(0.5, 4)):
    freqs, psd = signal.welch(epoch, sf)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx_band], freqs[idx_band])

# Extract 5 band power features: Delta, Theta, Alpha, Beta, Gamma
bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
features_ml = np.zeros((n_epochs, len(bands)))

for i in range(n_epochs):
    for j, (band_name, band_range) in enumerate(bands.items()):
        features_ml[i, j] = compute_bandpower(eeg_epochs[i], sf=100, band=band_range)

# Split data for ML
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(features_ml, labels, test_size=0.2, random_state=42)

# ----------- DEEP LEARNING PREPARATION ------------

def compute_spectrogram(epoch, sf=100, nperseg=128):
    freqs, times, Sxx = signal.spectrogram(epoch, sf, nperseg=nperseg)
    Sxx = Sxx / np.max(Sxx)  # normalize
    return Sxx

spectrograms = np.array([compute_spectrogram(epoch) for epoch in eeg_epochs])
spectrograms = spectrograms[..., np.newaxis]  # add channel dim

# One-hot encode labels for DL
labels_dl = to_categorical(labels, num_classes=n_classes)

# Train-test split DL
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(spectrograms, labels_dl, test_size=0.2, random_state=42)

# ----------- MACHINE LEARNING MODEL -----------------

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train_ml, y_train_ml)
y_pred_ml = clf_rf.predict(X_test_ml)
acc_ml = accuracy_score(y_test_ml, y_pred_ml)
print("ML (Random Forest) Accuracy: {:.3f}".format(acc_ml))
print(classification_report(y_test_ml, y_pred_ml))

# ----------- DEEP LEARNING MODEL (CNN) ---------------

model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train_dl.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax'),
])

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()

history = model_cnn.fit(X_train_dl, y_train_dl, epochs=15, batch_size=32, validation_split=0.1)

loss_dl, acc_dl = model_cnn.evaluate(X_test_dl, y_test_dl)
print(f"Deep Learning (CNN) Accuracy: {acc_dl:.3f}")

# ---------- COMPARISON PLOT -------------

methods = ['Machine Learning (RF)', 'Deep Learning (CNN)']
accuracies = [acc_ml, acc_dl]

plt.bar(methods, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Comparison of Sleep Pattern Classification Approaches')
plt.ylim(0, 1)
plt.show()
