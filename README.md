# Determining the level of difficulty of movies for English language learning
**This project aims** to develop a machine learning solution that can automatically determine the difficulty level of English-language movies, making it easier to choose movies that are both interesting and understandable. The input data consists of a labeled dataset with movie titles, subtitles, and language proficiency level tags (A2/B1/B2/C1). It is the task of **multiple classification**.

**The evaluation metric:** F1-score (macro) \
**The ML algorithm:** Stochastic Gradient Descent (SGD)

Before training, he text data was first preprocessed by removing unnecessary fragments and stopwords, lemmatizing the words, and also transforming the text into numerical features using *the CountVectorizer* and *TfidfTransformer*. Then the SGDClassifier algorithm was used with *GridSearchCV* to optimize the hyperparameters.

**The best achieved f1_score on the test set:** 0.632 \
**The best achieved f1_score on the train set:** 0.622

Furthermore, a [Streamlit application](https://veradics-english-score-app-ijccfp.streamlit.app/) was developed to facilitate the process of determining the level of difficulty for English language learning in movies by simply uploading an srt file.

### Files in directory:
- `english_score_notebook.ipynb` - notebook with data preprocessing and optimization ML model
- `app.py` - streamlit application file
- `sgd_model.pcl` - compressed model
- `movie_emage.jpg` - image for streamlit application

### Used libraries:
*Pandas, Numpy, Matplotlib, NLTK, SpaCy, Scikit-learn, Pickle, Streamlit*
