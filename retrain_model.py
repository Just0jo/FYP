import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- CONFIG ---
DATASET_PATH = "InstrumentData"  # Your instrument folder
MODEL_FILENAME = "new_instrument_classifier.pkl"  # New model name!

# --- FEATURE EXTRACTOR ---
def extract_features(file_path, n_mfcc=13, max_duration=7.0):
    y, sr = librosa.load(file_path, duration=max_duration, mono=True)
    y = y / np.max(np.abs(y))  # normalize

    # --- Features ---
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)  # 13
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)        # 12
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0) # 7

    # --- Combine ---
    combined = np.concatenate((mfcc, chroma, contrast))  # 13 + 12 + 7 = 32

    # --- Force to 38 Features ---
    EXPECTED_LENGTH = 38
    if len(combined) < EXPECTED_LENGTH:
        combined = np.pad(combined, (0, EXPECTED_LENGTH - len(combined)))
    else:
        combined = combined[:EXPECTED_LENGTH]

    return combined


# --- LOAD DATASET ---
features = []
labels = []

for instrument in os.listdir(DATASET_PATH):
    instrument_path = os.path.join(DATASET_PATH, instrument)
    if os.path.isdir(instrument_path):
        for file in os.listdir(instrument_path):
            if file.endswith(".wav"):
                file_path = os.path.join(instrument_path, file)
                try:
                    feat = extract_features(file_path)
                    features.append(feat)
                    labels.append(instrument)
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")

X = np.array(features)
y = np.array(labels)

# --- TRAIN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# --- SAVE THE MODEL ---
with open(MODEL_FILENAME, "wb") as f:
    pickle.dump(model, f)

print(f"💾 New model saved as {MODEL_FILENAME}")
