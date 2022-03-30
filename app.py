from flask import Flask, request, render_template
import joblib
import nltk
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

## List of words we need ##
#nltk.download('stopwords')
nltk.download('omw-1.4')
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

blacklist_vocab = ['able','also', 'another', 'anyone','based', 'cannot','could','else',
                   'every', 'following', 'found', 'get' 'getting', 'give', 'go', 'got',
                   'however', 'idea', 'instead', 'issue','know', 'help', 'last', 'let', 'like',
                   'look', 'make', 'need', 'next', 'none', 'one', 'please', 'possible',
                   'problem','question', 'react', 'say', 'see', 'seems', 'show', 'simple',
                   'solution', 'something','still','sure', 'thank', 'thanks',  'tried',
                   'try', 'trying', 'two', 'use', 'used',  'using', 'want', 'way', 'without',
                   'work', 'working', 'would', 'write', 'wrong']

bow = ['add', 'api', 'app', 'application', 'array', 'call', 'change', 'class',
       'code', 'column', 'com', 'console', 'const', 'content', 'create', 'data',
       'default', 'different', 'end', 'error', 'example', 'false', 'file', 'find',
       'first', 'function', 'get', 'getting', 'html', 'http', 'id', 'image', 'import',
       'index', 'input', 'int', 'json', 'key', 'line', 'list', 'log', 'main', 'message',
       'method', 'name', 'new', 'null', 'number', 'object', 'output', 'page', 'path', 'print',
       'project', 'public', 'python', 'read', 'request', 'result', 'return', 'row', 'run',
       'script', 'server', 'set', 'start', 'string', 'table', 'test', 'text', 'time', 'true', 'type',
       'url', 'user', 'value', 'var', 'version', 'work']

## Functions we need ##
lemmatizer = WordNetLemmatizer()

vectorizer = CountVectorizer(analyzer = "word", vocabulary = bow)

def message_to_words_v2(raw_title, raw_text):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Join Title and Texte
    join_text = " ".join((BeautifulSoup(raw_title).get_text(),BeautifulSoup(raw_text).get_text()))
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", join_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords+blacklist_vocab)
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Lemmatizing
    lemmat_text = [lemmatizer.lemmatize(w) for w in meaningful_words]
    #
    # 7. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join(lemmat_text))


@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier and labelencoder
        clf = joblib.load("clf.pkl")
        le = joblib.load("le.pkl")

        # Get values through input bars
        title = request.form.get("title")
        body = request.form.get("question")



        # Put inputs to dataframe
        X = vectorizer.fit_transform([message_to_words_v2(title, body)]).toarray()
        #pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])

        # Get prediction
        proba = clf.predict_proba(X)[0]
        #prediction = le.classes_[clf.predict(X)[0]]

        sorted_index_array = np.argsort(proba)

        # sorted array
        sorted_array = proba[sorted_index_array]

        # we want 2 largest value
        #n = 2

        #rslt = sorted_array[-n : ]

        prediction = [le.classes_[sorted_index_array[-1]],le.classes_[sorted_index_array[-2]]]

    else:
        prediction = ""

    return render_template("website.html", output = prediction)

if __name__ == "__main__":
    app.run(debug=True)
