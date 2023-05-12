# Determining the level of difficulty of movies for English language learning
**This project aims** to develop a machine learning solution that can automatically determine the difficulty level of English-language movies, making it easier to choose movies that are both interesting and understandable. The input data consists of a labeled dataset with movie titles, subtitles, and language proficiency level tags (A2/B1/B2/C1). It is the task of **multiple classification**.

**The evaluation metric:** F1-score (macro) \
**The ML algorithm:** Stochastic Gradient Descent (SGD)

Before training, he text data was first preprocessed by removing unnecessary fragments and stopwords, lemmatizing the words, and also transforming the text into numerical features using *the CountVectorizer* and *TfidfTransformer*. Then the SGDClassifier algorithm was used with *GridSearchCV* to optimize the hyperparameters.

**The best achieved f1_score on the test set:** 0.624 \
**The best achieved f1_score on the train set:** 0.622

### Files in directory:
- `english_score_notebook.ipynb` - notebook with data preprocessing and optimization ML model 

### Used libraries:
*Pandas, Numpy, Matplotlib, NLTK, SpaCy, Scikit-learn*
