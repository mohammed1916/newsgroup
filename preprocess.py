from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):
    # Preprocess the files
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ])

    preprocessed_data = pipeline.fit_transform(data.data)

    # Convert preprocessed_data to a format accepted by model.predict()
    preprocessed_data = preprocessed_data.toarray()

    return preprocessed_data

data = fetch_20newsgroups(subset='train', random_state=42)
preprocessed_data = preprocess_data(data)

print(preprocessed_data[0])
