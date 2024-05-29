import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize


def load_texts(base_dir: Path) -> Iterator[Tuple[str, List[str]]]:
    logging.info(f"Loading texts from {base_dir}")
    for txt_path in base_dir.glob("*/*.txt"):
        folder = txt_path.parent.name
        contents = txt_path.read_text()
        tokens = word_tokenize(contents)
        yield (folder, tokens)


def compute_features(
    data_loader: Iterator[Tuple[str, List[str]]],
    glove_model: KeyedVectors,
) -> List[Tuple[str, np.ndarray]]:
    logging.info("Computing GloVe features")
    res = []
    for folder, tokens in data_loader:
        embeddings = [glove_model[word] for word in tokens if word in glove_model]
        if embeddings:
            avg_pool = np.mean(embeddings, axis=0)
            max_pool = np.max(embeddings, axis=0)
            min_pool = np.min(embeddings, axis=0)
            feature_vector = np.concatenate(
                (avg_pool, max_pool, min_pool, [len(tokens)])
            )
            res.append((folder, feature_vector))
    logging.info("Done computing features")
    return res


def featurize_text(text: str, glove_model: KeyedVectors) -> Optional[np.ndarray]:
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
