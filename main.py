
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
import librosa
import pandas as pd
import csv
import io
import os
import sounddevice as sd
from datetime import datetime
from scipy.fft import fft, fftfreq
import pickle

with open("new_instrument_classifier.pkl", "rb") as f:
    instrument_model = pickle.load(f)

with open("chord_classifier.pkl", "rb") as f:
    simple_chord_model = pickle.load(f)

with open("full_chord_classifier.pkl", "rb") as f:
    full_chord_model = pickle.load(f)


def extract_features(file_path, max_duration=7.0):
    y, sr = librosa.load(file_path, mono=True, duration=max_duration)
    if len(y) < int(sr * max_duration):
        y = np.pad(y, (0, int(sr * max_duration) - len(y)), mode="constant")
    y = y / np.max(np.abs(y))


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)         # shape (20,)

    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)     # shape (12,)

    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)   # shape (6,)

    features = np.concatenate([mfcc_mean, chroma_mean, tonnetz_mean])
    return features

# --- APP SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sd.default.device = (7, None)
# --- FFT File Upload Endpoint ---
@app.post("/upload_fft")
async def upload_fft(file: UploadFile = File(...)):
    contents = await file.read()
    data, samplerate = sf.read(io.BytesIO(contents))

    if data.ndim > 1:
        data = data[:, 0] 

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)

    freqs = xf[:N // 2]
    magnitudes = 2.0 / N * np.abs(yf[:N // 2])

    peak_idx = np.argmax(magnitudes)
    peak_freq = freqs[peak_idx]
    peak_magnitude = magnitudes[peak_idx]

    return JSONResponse(content={
        "peak_frequency": round(peak_freq, 2),
        "peak_magnitude": round(peak_magnitude, 2)
    })



@app.post("/predict_chord_simple")
async def predict_chord_simple(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)

    features = extract_features("temp.wav")
    prediction = simple_chord_model.predict([features])[0]

    return {"predicted_chord": prediction}



@app.post("/predict_chord_full")
async def predict_chord_full(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)

    features = extract_features("temp.wav")
    prediction = full_chord_model.predict([features])[0]

    return {"predicted_chord": prediction}



@app.post("/instrument_tuner")
async def instrument_tuner(file: UploadFile = File(...)):
    contents = await file.read()
    y, sr = sf.read(io.BytesIO(contents))

    if y.ndim > 1:
        y = y[:, 0]  

    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1 / sr)
    positive_freqs = xf[:N // 2]
    magnitude = 2.0 / N * np.abs(yf[:N // 2])

    peak_idx = np.argmax(magnitude)
    peak_freq = positive_freqs[peak_idx]

    def frequency_to_note(freq):
        A4 = 440.0
        n = 12 * np.log2(freq / A4)
        semitone = int(round(n))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_idx = semitone % 12
        octave = 4 + (semitone // 12)
        return f"{notes[note_idx]}{octave}"

    closest_note = frequency_to_note(peak_freq)

    return {
        "peak_frequency": round(peak_freq, 2),
        "closest_note": closest_note
    }



@app.post("/instrument_key_detector")
async def instrument_key_detector(file: UploadFile = File(...)):
    contents = await file.read()
    with open("live_instrument.wav", "wb") as f:
        f.write(contents)

    features = extract_features("live_instrument.wav")
    instrument = instrument_model.predict([features])[0]

    y, sr = librosa.load("live_instrument.wav", duration=7.0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    most_active = np.argmax(chroma_mean)
    detected_key = note_names[most_active]

    # Save to CSV Log
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "prediction_log.csv"
    log_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["Timestamp", "Instrument", "Key", "Confidence (%)"])
        writer.writerow([now, instrument, detected_key, 95])  

    return {
        "instrument": instrument,
        "key": detected_key
    }



@app.get("/dashboard_logs")
async def dashboard_logs():
    log_file = "prediction_log.csv"
    if not os.path.exists(log_file):
        return {"error": "No logs found"}

    df = pd.read_csv(log_file)
    return df.to_dict(orient="records")



@app.get("/")
async def root():
    return {"message": "Backend is running"}
