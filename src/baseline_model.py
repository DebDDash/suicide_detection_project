from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def run_logreg(X_train, X_test, y_train, y_test, out_path="results/baseline_report.txt"):
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)

def run_svm(X_train, X_test, y_train, y_test, out_path="results/svm_report.txt"):
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
