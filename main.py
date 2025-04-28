from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
from io import BytesIO
import numpy as np
from scipy.fft import fft, fftfreq
import sounddevice as sd

app = FastAPI()

# Allow CORS for frontend communication
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
    # Read uploaded file
    contents = await file.read()
    data, samplerate = sf.read(BytesIO(contents))

    if data.ndim > 1:
        data = data[:, 0]  # Convert to mono if stereo

    # Perform FFT
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


# --- Real-time Audio Recording Endpoint --
@app.post("/record_live_audio")
async def record_live_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # Get the file content
        print(f"Received file with size: {len(contents)} bytes")

       

        # Use BytesIO to simulate file-like object in memory
        audio_file = BytesIO(contents)

        # Try to read the audio file
        
        data, samplerate = sf.read(audio_file)
        print(f"Audio data shape: {data.shape}, Sample rate: {samplerate}")

        if data.ndim > 1:  # Convert stereo to mono if needed
            data = data[:, 0]

        # Perform FFT on the audio data
        N = len(data)
        yf = fft(data)
        xf = fftfreq(N, 1 / samplerate)

        # Get frequencies and magnitudes for FFT
        freqs = xf[:N // 2]
        magnitudes = 2.0 / N * np.abs(yf[:N // 2])

        peak_idx = np.argmax(magnitudes)
        peak_freq = freqs[peak_idx]
        peak_magnitude = magnitudes[peak_idx]

        return {"peak_frequency": round(peak_freq, 2), "peak_magnitude": round(peak_magnitude, 2)}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}



# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Backend is running"}
