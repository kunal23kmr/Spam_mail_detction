import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib  # Import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("spam.csv", encoding="latin-1")

# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Features With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Save the model and CountVectorizer using joblib
joblib.dump(clf, 'spam-mail.pkl')
joblib.dump(cv, 'cv.pkl')
