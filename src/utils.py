# src/utils.py
import re
import numpy as np

def clean_text(s: str) -> str:
    """
    Pulizia base richiesta dalla traccia:
    - minuscole
    - rimozione punteggiatura (teniamo lettere/numeri/spazi)
    - compressione spazi
    """
    s = s.lower()
    s = re.sub(r"[^a-zàèéìòù0-9\s]", " ", s)   # punteggiatura -> spazio
    s = re.sub(r"\s+", " ", s).strip()
    return s

def combine_text(title: str, body: str) -> str:
    """
    Unisce title + body in un unico input testuale per il modello.
    """
    return clean_text(f"{title} {body}")

def top_influential_words(text: str, pipeline, top_k: int = 5):
    """
    Estrae le top-k parole/frasi (feature) che hanno contribuito di più
    alla classe predetta, per modelli lineari (LogisticRegression) su TF-IDF.

    Logica:
    - trasformo il testo con TF-IDF -> vettore x
    - prendo la classe predetta c
    - contribuzione di ogni feature = x_i * coef[c, i]
    - ordino e prendo le più alte (positive)
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = vectorizer.transform([clean_text(text)])
    pred_class = pipeline.predict([clean_text(text)])[0]
    class_index = list(clf.classes_).index(pred_class)

    # Feature names
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Contribuzioni
    coefs = clf.coef_[class_index]              # (n_features,)
    x = X.toarray().ravel()                     # (n_features,)
    contrib = x * coefs

    # Prendo solo feature presenti nel testo (x > 0) e contrib positive
    mask = (x > 0) & (contrib > 0)
    if not mask.any():
        return pred_class, []

    idx = np.argsort(contrib[mask])[::-1]
    words = feature_names[mask][idx][:top_k]
    scores = contrib[mask][idx][:top_k]

    return pred_class, list(zip(words.tolist(), scores.tolist()))
