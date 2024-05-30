from pathlib import Path

import gensim.downloader as api
import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import GLOVE_MODEL_SIZE, MODEL_DIR, OTHER_THRESHOLD
from utils import featurize_text
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Document(BaseModel):
    document_text: str

model = {}

def load_model():
    model_file = MODEL_DIR / "model.joblib"
    label_encoder_file = MODEL_DIR / "label_encoder.joblib"
    try:
        glove_model = api.load(GLOVE_MODEL_SIZE)    
        label_encoder = joblib.load(label_encoder_file)
        classifier = joblib.load(model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error loading model: {str(e)}')

    return glove_model, label_encoder, classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info('Server is loading model')
    glove_model, label_encoder, classifier = load_model()
    model['glove_model'] = glove_model
    model['label_encoder'] = label_encoder
    model['classifier'] = classifier
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/classify_document")
async def classify_text(doc: Document):
    features = featurize_text(doc.document_text, model['glove_model'])
    if not features:
        raise HTTPException(status_code=400, detail="Unable to featurize input text")

    probas = model['classifier'].predict_proba(features.reshape(1, -1))
    if np.max(probas) > OTHER_THRESHOLD:
        label = model['label_encoder'].inverse_transform(
            np.argmax(probas, keepdims=1).ravel()
        ).item()
        return {"label": label, "message": "Classification successful"}
    else:
        return {
            "label": "other",
            "message": "No strong match to pre-existing document classes",
        }
    
