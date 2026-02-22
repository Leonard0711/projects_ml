from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import io
import numpy as np

model = load_model("facial_recognition4.keras", compile=False)
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

app = FastAPI(title="Emotion Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

guide = {
    0: 'Molesto',
    1: "Disgusto",
    2: 'Miedo',
    3: 'Felicidad',
    4: 'Tristeza',
    5: 'Sorpresa',
    6: 'Neutral'
}

def preprocess(contents: bytes):
    image = load_img(io.BytesIO(contents), color_mode="grayscale", target_size=(48, 48))
    arr = img_to_array(image)
    return np.expand_dims(arr, axis=0).astype("float32")/255.0

def predict_sync(batch: np.array)->int:
    prediction = model.predict(batch, verbose=0)
    return int(np.argmax(prediction))

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    if file.content_type not in  {"image/png", "image/jpeg", "image/jpg", "image/jfif"}:
        raise HTTPException(status_code=400, detail="Formato no soportado")
    contents = await file.read()
    batch = await run_in_threadpool(preprocess, contents)
    emotion_index = await run_in_threadpool(predict_sync, batch)
    return {"emotion": guide[emotion_index]}