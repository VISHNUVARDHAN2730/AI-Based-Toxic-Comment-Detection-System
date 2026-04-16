import pandas as pd
import re
import nltk
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords

data = pd.DataFrame({
    "comment": [
        "I love this product",
        "You are stupid",
        "This is amazing",
        "I hate you",
        "Great job",
        "You idiot",
        "Well done",
        "This is terrible",
        "Awesome work",
        "Shut up fool"
    ],
    "label": [0,1,0,1,0,1,0,1,0,1]
})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Apply cleaning
data["cleaned"] = data["comment"].apply(clean_text)

X = data["cleaned"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

def predict_comment(comment):
    cleaned = clean_text(comment)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        return "🚫 Toxic Comment"
    else:
        return "✅ Normal Comment"

while True:
    user_input = input("\nEnter a comment (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result = predict_comment(user_input)
    print("Result:", result)