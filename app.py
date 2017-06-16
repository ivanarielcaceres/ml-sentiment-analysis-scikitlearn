from flask import Flask, render_template, request, jsonify
from translate import Translator
from langdetect import detect
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import langid
import textblob
import requests
import json 

app = Flask(__name__)
app.debug = True

#Configure MongoDB client
client = MongoClient('172.17.0.2', 27017)
db = client['twitterdb']
twitter_itaipu_db = db['sentiment_analysis']

def transl(text):
    to_lang = ['en', 'pt', 'es']
    response = {}
    try:
        from_lang =  textblob.TextBlob(text).detect_language()
        for lang in to_lang:
            translator = Translator(to_lang=lang, from_lang=from_lang)
            response[lang] = translator.translate(text)
        return json.dumps(response)
    except Exception as e:
        print(e)

def get_dataset(path):
    movie_reviews_data_folder = r'{}'.format(path)
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))
    return dataset

def build_pipeline():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])
    return pipeline

def train_model(pipeline, x_train, y_train):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search

def predict_test(grid_search):
    return grid_search.predict(x_test)

def evaluate(y_test, y_predicted, dataset):
    report = metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names)
    return report

def predict(grid_search, dataset, data):
    predict = grid_search.predict(data)
    results = []
    for sentence, category in zip(data, predict):            
        twitter_itaipu_db.insert({dataset.target_names[category]: sentence})
        results.append((sentence, dataset.target_names[category]))

    return results


@app.route('/healthz')
def healthz():
    return "it's alive"

@app.route('/translate', methods=['POST'])
def translate():
    params = request.get_json()
    to_lang = ['en', 'pt', 'es']
    text = params['text']
    response = transl(text)
    return response

@app.route('/sentiment', methods=['POST'])
def sentiment():
    params = request.get_json()
    pipeline = build_pipeline()
    path = '/var/datasets/nltk_data/corpora/movie_reviews'
    dataset = get_dataset(path)
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)
    grid_search = train_model(pipeline, x_train, y_train)
    print(grid_search)
    results = []
    for sentence in params:
        sentence['text'] = sentence['text'].encode('ascii', 'ignore').decode('ascii')
        translated = transl(sentence['text'])
        results.append(translated)
    prediction = predict(grid_search, dataset, results)
    #twitter_itaipu_db.insert(dict((y, x) for x, y in prediction))
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
