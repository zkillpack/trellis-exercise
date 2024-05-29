from pathlib import Path

import gensim.downloader as api
import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import GLOVE_MODEL_SIZE, MODEL_DIR, OTHER_THRESHOLD
from utils import featurize_text

app = FastAPI()

model_file = MODEL_DIR / "model.joblib"
label_encoder_file = MODEL_DIR / "label_encoder.joblib"

glove_model = api.load(GLOVE_MODEL_SIZE)
model = joblib.load(model_file)
label_encoder = joblib.load(label_encoder_file)


class Document(BaseModel):
    document_text: str


@app.post("/classify_document")
def classify_text(doc: Document):
    features = featurize_text(doc.document_text, glove_model)
    if features is not None:
        probas = model.predict_proba(features.reshape(1, -1))
        if np.max(probas) > OTHER_THRESHOLD:
            label = label_encoder.inverse_transform(
                np.argmax(probas, keepdims=1).ravel()
            ).item()
            return {"label": label, "message": "Classification successful"}
        else:
            return {
                "label": "other",
                "message": "No strong match to pre-existing document classes",
            }
    else:
        raise HTTPException(status_code=400, detail="bad thing happen!")
