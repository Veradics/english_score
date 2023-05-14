import streamlit as st
import pysrt
import re
import pickle
import spacy

import nltk
from nltk.corpus import stopwords as nltk_stopwords


# load the trained model
def load():
    with open('./sgd_model.pcl', 'rb') as fid:
        return pickle.load(fid)

# image
st.image('./movie_image.jpg', caption=' Lights, camera, action!')

# heading
st.header('Find out the level of English in the movie!')

# loading file
srt_file = st.file_uploader("Upload a subtitle file in SRT format:", type="srt")


# PREPROCESSING
# stopwords for English from nltk
nltk.download('stopwords')
stopwords = nltk_stopwords.words('english')

# additional stopwords
pronouns = ['I', 'me', 'you', 'he', 'she', 'it']
articles = ['a', 'an', 'the']
prepositions = ['in', 'on', 'at', 'by', 'to', 'for', 'from']
conjunctions = ['and', 'but', 'or', 'yet']
interjections = ['oh', 'shh', 'yeah', 'ah', 'uh', 'mmm']

stopwords = stopwords + pronouns + articles + prepositions + conjunctions + interjections

def clean_stopwords(text):
    text = [word for word in text if word not in stopwords]
    return " ".join(text)


# lemmatization
nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]

    return lemmas


# regular expressions
HTML = r'<.*?>'  # html tags
TAG = r'{.*?}'  # other tags
COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]'  # comments
UPPER = r'[[A-Za-z ]+[\:\]]'  # speakers names (BOBBY:)
LETTERS = r'[^a-zA-Z\'.,!? ]'  # need only the letters
APOSTROPHES = r' [A-Za-z]+\'[A-Za-z]+ '  # delete words with apostrophes
SPACES = r'([ ])\1+'  # multiple spaces
SYMB = r"[^\w\d'\s]"  # puctuation marks


def clean_text(text):
    """
    Clean up the text by removing unnecessary elements.
    """
    # replace with a space
    for element in [HTML, TAG, COMMENTS, UPPER, LETTERS, APOSTROPHES]:
        text = re.sub(element, ' ', text)

        # replace with an empty string
    text = re.sub(SYMB, '', text)
    text = re.sub('www', '', text)
    text = re.sub("''", '', text)

    # spaces
    text = re.sub(SPACES, r'\1', text)

    # delete non-ascii characters
    text = text.encode('ascii', 'ignore').decode()

    text = text.strip()  # delete left and right spaces
    text = text.lower()  # lowercase

    return text


def prepare_srt(srt_file):
    """
    Prepare subtitles text for model.
    """
    srt_text = srt_file.read().decode('iso-8859-1')
    subs = pysrt.from_string(srt_text)
    subs = ' '.join([sub.text for sub in subs])
    subs = clean_text(subs)      # delete incorrect characters
    subs = lemmatize_text(subs)  # lemmatization
    subs = clean_stopwords(subs) # delete stopwords

    return subs

if srt_file is not None:
    # applying preprocessing
    subs = prepare_srt(srt_file)

    # prediction
    model = load()
    y_predict = model.predict([subs])

    st.subheader(f'Predicted English level: {y_predict[0]}')
