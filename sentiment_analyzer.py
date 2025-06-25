import pandas as pd
import string 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'text' : [
        'I love this product!',
        'This is the worst experience ever.',
        'Absolutely fantastic Service',
        'I am not happy with the quality',
        'Great value for money.',
        'Terrible customer support.'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['clean_text'])

# Show TF-IDF matrix
print("\nTF-IDF Matrix (x.toarray()):")
print(x.toarray())

# Show feature names (words)
print("\nFeature Names (Words):")
print(vectorizer.get_feature_names_out())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, df['sentiment'], test_size=0.3, random_state=42)

# Print what's inside
print("\nX_train (TF-IDF numbers for training sentences):")
print(X_train.toarray())

print("\ny_train (Sentiment labels for training sentences):")
print(y_train.tolist())

print("\nX_test (TF-IDF numbers for testing sentences):")
print(X_test.toarray())

print("\ny_test (Sentiment labels for testing sentences):")
print(y_test.tolist())

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
print("\nPredictions:", y_pred.tolist())
print("Actual:", y_test.tolist())

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
