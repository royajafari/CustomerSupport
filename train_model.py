import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
data = {
    'text': [
        'I love this product!',
        'This is the worst thing I have ever bought.',
        'Absolutely fantastic! Highly recommend.',
        'I am not happy with this purchase.',
        'It was okay, not great but not bad.',
        'I will never buy this again.',
        'Totally worth the money!',
        'Terrible customer service.',
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral',
        'negative',
        'positive',
        'negative',
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(df['text'], df['label'])

# Save the model to a file
joblib.dump(model, 'text_classifier.pkl')

print("Model trained and saved as 'text_classifier.pkl'.")
