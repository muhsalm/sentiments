import re
import spacy
from spacy.cli import download
from sklearn.base import TransformerMixin, BaseEstimator

# Ensure the model is downloaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    # reduce multiple spaces and newlines to only one
    text = re.sub(r'(\s\s+|\n\n+)', r' ', text)
    # remove double quotes
    text = re.sub(r'"', '', text)
    return text

def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())
    text = ' '.join(tokens)
    return text

class preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
