import shap
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import numpy as np

def explain_with_shap(model, X_test, vectorizer, out_path="results/shap_summary.png"):
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        shap.summary_plot(
            shap_values,
            features=X_test,
            feature_names=vectorizer.get_feature_names_out(),
            show=False
        )
        plt.savefig(out_path)
        print(f"SHAP summary saved to {out_path}")
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")

def explain_with_lime(model, vectorizer, text_sample):
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=["non suicide", "depression", "suicide"]
    )
    predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))
    exp = explainer.explain_instance(text_sample, predict_fn, num_features=10)
    return exp.as_list()
