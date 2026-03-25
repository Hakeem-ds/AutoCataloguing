from typing import List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, vectorizer, le, X_texts: List[str], y_true_labels: List[str]):
    X_vec = vectorizer.transform(X_texts)
    safe_labels = [lbl if lbl in set(le.classes_) else le.classes_[0] for lbl in y_true_labels]
    y_true_enc = le.transform(safe_labels)
    y_pred_enc = model.predict(X_vec)
    rep = classification_report(y_true_enc, y_pred_enc, target_names=le.classes_, output_dict=True)
    conf = confusion_matrix(y_true_enc, y_pred_enc, labels=list(range(len(le.classes_))))
    acc = accuracy_score(y_true_enc, y_pred_enc)
    macro_f1 = f1_score(y_true_enc, y_pred_enc, average="macro")
    return {
        "report": rep,
        "confusion_matrix": conf,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
