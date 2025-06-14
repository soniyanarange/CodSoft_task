import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("c:/Users/SONIA/OneDrive/Desktop/task 1/movie_genre_dataset_clean.csv")

# Step 2: Check available columns
print("Available columns:", df.columns)
import pandas as pd

# Sample data manually create karte hain
data = {
    "plot": [
        "A young man goes on a journey to find treasure and love.",
        "A spaceship crew explores a new galaxy and meets aliens.",
        "A detective investigates a mysterious murder case in town.",
        "A group of friends go on an adventure in the mountains.",
        "A hero saves the world from destruction.",
        "Romantic couple falls in love during college.",
        "Detectives investigate a mysterious murder.",       
        "Woman finds love in Paris,Romance.",
        "A haunted house terrifies its visitors.",
        "Spies go undercover to stop a nuclear war.",
        "Boy and girl from different worlds fall in love.",
        "Group of friends trapped in a forest with killer.",
        "Police chase a cunning serial killer.",
    ],
    "genre": [
        "Romance",
        "Sci-Fi",
        "Mystery",
        "Adventure",
        "Action",
        "Romance",
        "Thriller",
        "Romance",
        "Horrer",
        "Action",
        "Romance",
        "Horrer",
        "Thriller"
    ]
}

# DataFrame create karo
df = pd.DataFrame(data)

# CSV file save karo
df.to_csv("movie_genre_dataset_clean.csv", index=False)

print("Dataset created successfully ✅")

# Step 3: Clean text function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()                          # lowercase
    text = re.sub(r'\d+', '', text)              # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()                          # remove leading/trailing whitespaces
    return text

# Step 4: Clean the 'plot' column
df['clean_description'] = df['plot'].apply(clean_text)

# Step 5: Features (X) and Labels (y)
X = df['clean_description']
y = df['genre']

# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Step 7: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Step 8: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")