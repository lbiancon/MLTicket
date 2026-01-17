# src/train_eval.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

from utils import combine_text

def build_pipeline():
    """
    Pipeline minimale e standard per text classification:
    TF-IDF (unigrammi + bigrammi) + Logistic Regression.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

def train_and_evaluate(csv_path="data/tickets_sintetici.csv"):
    df = pd.read_csv(csv_path)

    X = [combine_text(t, b) for t, b in zip(df["title"], df["body"])]

    # Modello categoria
    y_cat = df["category"].astype(str).tolist()
    # allena il train 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.20, random_state=7, stratify=y_cat
    )

    cat_model = build_pipeline()
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)

    print("\n=== MODELLO CATEGORIA ===")
    print(f"Accuracy: {accuracy_score(y_test, cat_pred):.3f}")
    print(f"F1 macro: {f1_score(y_test, cat_pred, average='macro'):.3f}\n")
    print(classification_report(y_test, cat_pred, digits=3))

    cm = confusion_matrix(y_test, cat_pred, labels=cat_model.named_steps["clf"].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.named_steps["clf"].classes_)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix - Categoria")
    plt.tight_layout()
    plt.savefig("category_confusion_matrix.png", dpi=150)
    plt.close()

    # Modello priorità (etichette generate da regole, ma qui le “impariamo” per richiesta traccia)
    y_pri = df["priority"].astype(str).tolist()
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_pri, test_size=0.20, random_state=7, stratify=y_pri
    )

    pri_model = build_pipeline()
    pri_model.fit(X_train2, y_train2)
    pri_pred = pri_model.predict(X_test2)

    print("\n=== MODELLO PRIORITÀ ===")
    print(f"Accuracy: {accuracy_score(y_test2, pri_pred):.3f}")
    print(f"F1 macro: {f1_score(y_test2, pri_pred, average='macro'):.3f}\n")
    print(classification_report(y_test2, pri_pred, digits=3))

    cm2 = confusion_matrix(y_test2, pri_pred, labels=pri_model.named_steps["clf"].classes_)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=pri_model.named_steps["clf"].classes_)
    disp2.plot(values_format="d")
    plt.title("Confusion Matrix - Priorità")
    plt.tight_layout()
    plt.savefig("priority_confusion_matrix.png", dpi=150)
    plt.close()

    # Grafico semplice: conteggi classi (utile per report)
    df["category"].value_counts().plot(kind="bar")
    plt.title("Distribuzione classi - Categoria")
    plt.tight_layout()
    plt.savefig("category_distribution.png", dpi=150)
    plt.close()

    df["priority"].value_counts().plot(kind="bar")
    plt.title("Distribuzione classi - Priorità")
    plt.tight_layout()
    plt.savefig("priority_distribution.png", dpi=150)
    plt.close()

    # Salvataggio modelli
    joblib.dump(cat_model, "models/category_model.joblib")
    joblib.dump(pri_model, "models/priority_model.joblib")
    print("\nSalvati: models/category_model.joblib e models/priority_model.joblib")
    print("Grafici salvati: category_confusion_matrix.png, priority_confusion_matrix.png, category_distribution.png, priority_distribution.png")

def predict_batch(csv_in: str, csv_out: str):
    """
    Legge un CSV con colonne id,title,body e produce predizioni.
    """
    import joblib
    from utils import combine_text

    cat_model = joblib.load("models/category_model.joblib")
    pri_model = joblib.load("models/priority_model.joblib")

    df = pd.read_csv(csv_in)
    X = [combine_text(t, b) for t, b in zip(df["title"], df["body"])]

    df["pred_category"] = cat_model.predict(X)
    df["pred_priority"] = pri_model.predict(X)

    df.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"Creato: {csv_out}")

if __name__ == "__main__":
    train_and_evaluate("data/tickets_sintetici.csv")
