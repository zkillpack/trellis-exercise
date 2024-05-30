import contextlib
import logging
import os

import gensim.downloader as api
import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import BASE_MODEL_PARAMS, DATA_DIR, GLOVE_MODEL_SIZE, MODEL_DIR
from utils import compute_features, load_texts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Loading GloVe model")
with open(os.devnull, "w") as null:
    with contextlib.redirect_stdout(null):  # Gensim downloader is very loud
        glove_model = api.load(GLOVE_MODEL_SIZE)
logging.info("GloVe model loaded")

data_loader = load_texts(DATA_DIR)
features = compute_features(data_loader, glove_model)

df = pd.DataFrame(features, columns=["label", "features"])
df = df[df["label"] != "other"]
X = pd.DataFrame(df["features"].tolist())
y = df["label"]

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, random_state=12345
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=12345
)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_val_encoded = label_encoder.transform(y_val)

train_data = lgb.Dataset(X_train, label=y_train_encoded)


def objective(trial):
    global model
    trial_params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    trial_params |= BASE_MODEL_PARAMS

    model = lgb.LGBMClassifier(**trial_params)
    model.fit(X_train, y_train_encoded)
    return f1_score(y_test_encoded, model.predict(X_test), average="macro")


def keep_best_model(study, trial):
    global best_model
    if study.best_trial == trial:
        best_model = model


logging.info("Training model")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, callbacks=[keep_best_model])
logging.info("Model training completed")

y_preds = best_model.predict(X_val)
accuracy = accuracy_score(y_val_encoded, y_preds)
logging.info(f"Best model's validation accuracy: {accuracy:.4f}")
logging.info(
    f"Best model's validation P&R stats:\n{classification_report(y_val_encoded, y_preds)}"
)

joblib.dump(best_model, MODEL_DIR / "model.joblib")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

logging.info("Model saved successfully")
