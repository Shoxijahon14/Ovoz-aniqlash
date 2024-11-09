
import librosa
import numpy as np
import pyaudio
from sklearn.svm import SVC
import os


# Ovoz xususiyatlarini olish funksiyasi
def extract_features(audio_data, sr):
    pitch = librosa.yin(audio_data.astype(np.float32), fmin=75, fmax=300, sr=sr)
    avg_pitch = np.mean(pitch) if pitch.size > 0 else 0  # Agar pitch bo'sh bo'lsa, 0 qiymat qaytadi
    return [avg_pitch]


# Datasetdan olingan xususiyatlarni va belgilarni yuklash
def load_dataset():
    X = []  # Xususiyatlar
    y = []  # Klassifikatsiya belgilari (erkak - 0, ayol - 1)

    # Erkak ovozlarini yuklash
    for file in os.listdir("data/erkak"):
        if file.endswith(".wav"):
            audio_data, sr = librosa.load(os.path.join("data/erkak", file), sr=None)
            features = extract_features(audio_data, sr)
            X.append(features)
            y.append(0)  # Erkak

    # Ayol ovozlarini yuklash
    for file in os.listdir("data/ayol"):
        if file.endswith(".wav"):
            audio_data, sr = librosa.load(os.path.join("data/ayol", file), sr=None)
            features = extract_features(audio_data, sr)
            X.append(features)
            y.append(1)  # Ayol

    return X, y


# Modelni o'qitish
def train_model(X, y):
    model = SVC(kernel="linear")
    model.fit(X, y)
    return model


# Ovoz faylini tasniflash
def classify_voice(audio_data, sr, model):
    features = extract_features(audio_data, sr)
    features = np.array(features).reshape(1, -1)  # 1D dan 2D ga o'zgartirish
    prediction = model.predict(features)
    if prediction == 0:
        return "Erkak"
    else:
        return "Ayol"


# Mikrofondan ovoz olish
def record_audio(duration=3, sample_rate=22050):
    p = pyaudio.PyAudio()

    # Mikrofon sozlamalari
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    frames = []

    # Audio yozish
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Raw audio data-ni numpy array ga aylantirish va float32 formatiga o'zgartirish
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)

    return audio_data, sample_rate


# Datasetdan ma'lumotlarni yuklash va modelni o'qitish
X, y = load_dataset()
model = train_model(X, y)

# Mikrofondan ovoz olish va tasniflash
if model:
    # Yangi ovoz yozib olish
    audio_data, sr = record_audio(duration=3)

    # Yozilgan ovoz tasnifi
    natija = classify_voice(audio_data, sr, model)
    print("Tasniflash natijasi:", natija)
else:
    print("Modelni o'qitish uchun dataset yuklanmadi.")
