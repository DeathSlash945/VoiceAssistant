import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm

DATASET_PATH = "voiceData"
FEATURES_PATH = "voiceData/features"
os.makedirs(FEATURES_PATH, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time, n_mfcc)

def process_folder(folder, label):
    features = []
    labels = []
    for file in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            mfcc = extract_mfcc(filepath)
            features.append(mfcc)
            labels.append(0 if label == "randomNoise" else 1)
    return features, labels

def save_data():
    wake_feat, wake_labels = process_folder(os.path.join(DATASET_PATH, "wakeWord"), "wakeWord")
    notwake_feat, notwake_labels = process_folder(os.path.join(DATASET_PATH, "randomNoise"), "randomNoise")

    # Pad or truncate to same length
    max_len = 50
    def pad(x): return x[:max_len] if len(x) >= max_len else np.pad(x, ((0, max_len - len(x)), (0, 0)))

    X = np.array([pad(x) for x in wake_feat + notwake_feat])
    y = np.array(wake_labels + notwake_labels)

    np.save(os.path.join(FEATURES_PATH, "X.npy"), X)
    np.save(os.path.join(FEATURES_PATH, "y.npy"), y)

if __name__ == "__main__":
    save_data()
