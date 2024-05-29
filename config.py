from pathlib import Path

DATA_DIR = Path("./trellis_assessment_ds/")
MODEL_DIR = Path("./model/")

GLOVE_MODEL_SIZE = "glove-wiki-gigaword-50"

BASE_MODEL_PARAMS = {
    "objective": "multiclassova",
    "metric": "multi_logloss",
    "verbose": -1,
}
