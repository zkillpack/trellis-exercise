from pathlib import Path

# Relative paths to store model and data
DATA_DIR = Path("./data/")
MODEL_DIR = Path("./model/")

# Gensim model from https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models
GLOVE_MODEL_SIZE = "glove-wiki-gigaword-50"

# Probability threshold below which we return 'other'
OTHER_THRESHOLD = 0.95

# LightGBM params
BASE_MODEL_PARAMS = {
    "objective": "multiclassova",
    "metric": "multi_logloss",
    "verbose": -1,
}
