from pathlib import Path

DATA_DIR = Path("./data/")
MODEL_DIR = Path("./model/")

GLOVE_MODEL_SIZE = "glove-wiki-gigaword-50"

BASE_MODEL_PARAMS = {
    "objective": "multiclassova",
    "metric": "multi_logloss",
    "verbose": -1,
}
