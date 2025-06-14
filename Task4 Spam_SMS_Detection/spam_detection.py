#1. Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

#2. Load Dataset

#Use a dataset like the SMS Spam Collection Dataset.

df = pd.read_csv('spam.csv', encoding='latin-1')
df.columns = ['label', 'message']

#3. Preprocess Data

#Convert labels and split the dataset.

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Vectorization (TF-IDF)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#5. Train Classifiers

#A. Naive Bayes

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

#B. Logistic Regression

lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

#C. Support Vector Machine

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

#6. Evaluate Models

models = {'Naive Bayes': nb, 'Logistic Regression': lr, 'SVM': svm}

for name, model in models.items():
    preds = model.predict(X_test_tfidf)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))