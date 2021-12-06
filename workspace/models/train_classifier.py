import sys
import re
import pickle
import nltk
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


stop_words = set(stopwords.words())

nltk.download(['punkt', 'stopwords', 'wordnet'])

def load_data(database_filepath):

    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 6:]
    return X, Y, Y.columns


def tokenize(text):

    
    # Lower case the text
    text = text.lower()

    # Remove punctuation
    text = re.sub("[^a-zA-Z0-9]", " ", text)

    # Tokenize the text
    text = word_tokenize(text)

    # Remove stop words
    text = [word for word in text if word not in stop_words]

    # Lemmatize the text
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    return text

def build_model():

    
    # pipeline
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', OneVsRestClassifier(LinearSVC()))
    ])
    
    #parameters  
    parameters = {
        'classifier__estimator__C': (0.1, 1, 10)
    }


    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
  
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    


def save_model(model, model_filepath):
    
    filename = 'classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()