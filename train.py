import json
import pickle
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('punkt_tab')

# Paths
BASE_DIR = os.path.dirname(__file__)
intents_path = os.path.join(BASE_DIR, 'dataset', 'intents1.json')
model_path = os.path.join(BASE_DIR, 'model', 'chatbot_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

with open(intents_path, 'r') as f:
    intents = json.load(f)

text_data = []
labels = []

for intent in intents['intents']:
    for example in intent['patterns']:
        text_data.append(example.lower())
        labels.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
y = labels

model = DecisionTreeClassifier()
model.fit(X, y)

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model retrained and saved successfully!")
