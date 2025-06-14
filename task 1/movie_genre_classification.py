import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset
df = pd.read_csv("movie_genre_dataset.csv")  # Make sure file name is correct

# Step 2: Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_description'] = df['description'].apply(clean_text)

# Step 3: Remove rare genres (only 1 sample)
genre_counts = df['genre'].value_counts()
df = df[df['genre'].isin(genre_counts[genre_counts >= 2].index)]

# Step 4: Encode genre
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Step 5: TF-IDF feature extraction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['clean_description'])
y = df['genre_encoded']

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

unique_labels = sorted(set(y_test) | set(y_pred))
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(unique_labels)))

# Step 9: Predict new plot
def predict_genre(plot):
    cleaned = clean_text(plot)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)
    return le.inverse_transform(pred)[0]

# Example usage
# print(predict_genre("A team of scientists explores a mysterious alien planet."))
