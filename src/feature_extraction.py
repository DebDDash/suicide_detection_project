from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(train_texts, test_texts, max_features=20000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
