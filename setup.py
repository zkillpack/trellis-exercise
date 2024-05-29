import logging

import gensim.downloader as api
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import BASE_MODEL_PARAMS, DATA_DIR, GLOVE_MODEL_SIZE, MODEL_DIR
from utils import compute_features, load_texts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


logging.info("Loading GloVe model")
glove_model = api.load(GLOVE_MODEL_SIZE)
logging.info("GloVe model loaded")

data_loader = load_texts(DATA_DIR)
features_data = compute_features(data_loader, glove_model)

df = pd.DataFrame(features_data, columns=["label", "features"])
df = df[df["label"] != "other"]
X = pd.DataFrame(df["features"].tolist())
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

train_data = lgb.Dataset(X_train, label=y_train_encoded)

params = BASE_MODEL_PARAMS
params["num_class"] = len(label_encoder.classes_)

model = lgb.train(params, train_data, 1000)
logging.info("Model training completed")


MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_model(MODEL_DIR / "model.txt")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

logging.info("Model saved successfully")
