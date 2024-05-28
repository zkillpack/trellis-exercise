import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_texts(data_dir):
    logging.info(f"Loading texts from {data_dir}")
    base_dir = Path(data_dir)
    for txt_path in base_dir.glob('*/*.txt'):
        folder = txt_path.parent.name
        contents = txt_path.read_text()
        tokens = word_tokenize(contents)
        yield (folder, tokens)

def compute_features(data_loader, glove_model):
    logging.info('Computing GloVe features')
    res = []
    for folder, tokens in tqdm(data_loader):
        embeddings = [glove_model[word] for word in tokens if word in glove_model]
        if embeddings:  
            avg_pool = np.mean(embeddings, axis=0)
            max_pool = np.max(embeddings, axis=0)
            min_pool = np.min(embeddings, axis=0)
            feature_vector = np.concatenate((avg_pool, max_pool, min_pool, [len(tokens)]))
            res.append((folder, feature_vector))
    logging.info('Done computing features')
    return res

logging.info('Loading GloVe model')
glove_model = api.load("glove-wiki-gigaword-50")
logging.info('GloVe model loaded')

data_dir = './trellis_assessment_ds'  
data_loader = load_texts(data_dir)
features_data = compute_features(data_loader, glove_model)

df = pd.DataFrame(features_data, columns=['label', 'features'])
df = df[df['label'] != 'other']
X = pd.DataFrame(df['features'].tolist())
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


params = {
    'objective': 'multiclassova',
    'num_class': len(label_encoder.classes_),
    'metric': 'multi_logloss',
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train_encoded)
model = lgb.train(params, train_data, 1000)
logging.info("Model training completed")

Path('./model/').mkdir(parents=True, exist_ok=True)
model.save_model('./model/model.txt')
logging.info("Model saved successfully")