from src.feature_extraction import get_tfidf
from src.baseline_model import run_logreg, run_svm
from src.transformer_model import run_transformer
from src.explainability import explain_with_shap, explain_with_lime
from src.preprocess import prepare_baseline_data

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = prepare_baseline_data("data/Suicide_Detection.csv")
    print(f"Dataset ready: {len(X_train)} train, {len(X_test)} test")

    # Baselines
    print("Extracting TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = get_tfidf(X_train, X_test)

    print("Running Logistic Regression baseline...")
    run_logreg(X_train_tfidf, X_test_tfidf, y_train, y_test)

    print("Running SVM baseline...")
    run_svm(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Explain baseline
    from sklearn.linear_model import LogisticRegression
    print("Fitting explainability model (LogReg)...")
    logreg = LogisticRegression(max_iter=500).fit(X_train_tfidf, y_train)

    print("Running SHAP explainability...")
    explain_with_shap(logreg, X_test_tfidf, vectorizer)

    # Transformers
    print("Running BERT transformer...")
    run_transformer(X_train, X_test, y_train, y_test, model_name="bert-base-uncased")

    print("Running RoBERTa transformer...")
    run_transformer(X_train, X_test, y_train, y_test, model_name="roberta-base")

    # LIME example
    sample = X_test.iloc[0]
    print("Running LIME explanation...")
    print("LIME Explanation:", explain_with_lime(logreg, vectorizer, sample))

if __name__ == "__main__":
    main()
