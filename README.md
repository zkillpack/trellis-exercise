To train the model and launch the server, run:

`docker compose --progress plain up  --build  2>&1 | tee build.log

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
     -d '{"document_text": "I invented a new kind of hot sauce taco that is delicious."}'
{"label":"food","message":"Classification successful"}
```

Texts where none of the one-vs-all classifiers have a probability estimate above OTHER_THRESHOLD will be assigned the 'other' label:

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": "Strange rainbow underfoot keyboard elves what in the world happened to the mail sewing machine swedish snus."}'
{"label":"other","message":"No strong match to pre-existing document classes"}
```

Documents that can't be featurized will return HTTP 400:

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": "awf8e9y39ryandsfjksf"}'
{"detail":"Unable to featurize input text"}
```

Input that does not contain a `document_text` field, or input that contains a `document_text` that is not a string with at least one character will return HTTP 422:

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document": "Hello"}'

{"detail":[{"type":"missing","loc":["body","document_text"],"msg":"Field required","input":{"document":"Hello"}}]}
```

```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": ""}'

{"detail":[{"type":"string_too_short","loc":["body","document_text"],"msg":"String should have at least 1 character","input":"","ctx":{"min_length":1}}]}
```


```
curl -X POST "http://localhost:8000/classify_document" \
     -H "Content-Type: application/json" \
     -d '{"document_text": 42}'

{"detail":[{"type":"string_type","loc":["body","document_text"],"msg":"Input should be a valid string","input":42}]}
```