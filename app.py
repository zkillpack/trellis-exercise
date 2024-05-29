from pathlib import Path

import gensim.downloader as api
import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import GLOVE_MODEL_SIZE, MODEL_DIR
from utils import featurize_text

app = FastAPI()

model_file = MODEL_DIR / "model.txt"
label_encoder_file = MODEL_DIR / "label_encoder.joblib"

glove_model = api.load(GLOVE_MODEL_SIZE)
model = lgb.Booster(model_file=str(model_file))
label_encoder = joblib.load(label_encoder_file)


class Document(BaseModel):
    document_text: str


@app.post("/classify_document/")
def classify_text(doc: Document):
    features = featurize_text(doc.document_text, glove_model)
    if features is not None:
        probas = model.predict(features.reshape(1, -1))
        label = label_encoder.inverse_transform(
            np.argmax(probas, keepdims=1).ravel()
        ).item()
        return {"label": label, "message": "Classification successful"}
    else:
        raise HTTPException(status_code=400, detail="bad thing happen")
