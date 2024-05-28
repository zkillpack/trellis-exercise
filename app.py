from pathlib import Path

import gensim.downloader as api
import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import word_tokenize
from pydantic import BaseModel

app = FastAPI()

glove_model = api.load("glove-wiki-gigaword-50")
model = lgb.Booster(model_file="./model/model.txt")
label_encoder = joblib.load("./model/label_encoder.joblib")


class Document(BaseModel):
    document_text: str


def featurize_text(text):
    tokens = word_tokenize(text)
    embeddings = [glove_model[word] for word in tokens if word in glove_model]
    if embeddings:
        avg_pool = np.mean(embeddings, axis=0)
        max_pool = np.max(embeddings, axis=0)
        min_pool = np.min(embeddings, axis=0)
        feature_vector = np.concatenate((avg_pool, max_pool, min_pool, [len(tokens)]))
        return feature_vector
    else:
        return None


@app.post("/classify/")
def classify_text(doc: Document):
    features = featurize_text(doc.document_text)
    if features is not None:
        probas = model.predict(features.reshape(1, -1))
        label = label_encoder.inverse_transform(
            np.argmax(probas, keepdims=1).ravel()
        ).item()
        return {"label": label, "message": "Classification successful"}
    else:
        raise HTTPException(status_code=400, detail="bad thing happen")
