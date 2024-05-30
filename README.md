To train the model and launch the server, run:

`docker compose up --build`

This will:

* Set up Python + packages and build LightGBM
* Download nltk's 'punkt' tokenizer + GloVe vectors 
* Extract the sample data and train a simple classifier using hyperparameter tuning
* Log model metrics to the Docker build log (accuracy, precision, recall, F1 score)

The trained model will be built into the image, so subsequent runs can be invoked via

`docker compose up`

To test the model:

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": "Strange rainbow underfoot keyboard elves what in the world happened to the mail sewing machine swedish snus."}'
{"label":"other","message":"No strong match to pre-existing document classes"}
```

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": "I invented a new kind of hot sauce taco that is delicious."}'
{"label":"food","message":"Classification successful"}
```

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": "awf8e9y39ryandsfjksf"}'
{"detail":"Unable to featurize input text"}
```