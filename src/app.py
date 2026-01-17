# src/app.py
import streamlit as st
import pandas as pd
import joblib

from utils import combine_text, top_influential_words, clean_text

# st.set_page_config(page_title="MLTicket", layout="centered")
# provato a mettere wide per avere piu' leggibilita' sia dei grafici che degli elementi 

st.set_page_config(page_title="MLTicket", layout="wide")

@st.cache_resource

def load_models():
    cat_model = joblib.load("models/category_model.joblib")
    pri_model = joblib.load("models/priority_model.joblib")
    return cat_model, pri_model

cat_model, pri_model = load_models()

st.title("Triage automatico MLTicket (ML minimale)")
st.write("Inserisci un ticket e ottieni: **categoria**, **priorità**, e **5 parole più influenti**.")

import random

st.subheader("1) Predizione su singolo ticket")

# Esempi rapidi (automotive / aziendali) per dimostrazione
ESEMPI = {
    "Seleziona un esempio...": ("", ""),
    "Tecnico - Errore 500 (alta)": (
        "Errore 500 su dashboard manutenzione",
        "Da stamattina errore 500 quando apro la schermata principale, situazione urgente e bloccante."
    ),
    "Tecnico - Linea ferma (alta)": (
        "Blocco postazione OBD dopo aggiornamento",
        "Dopo aggiornamento software la postazione OBD non si avvia e la linea si ferma. Serve intervento urgente."
    ),
    "Amministrazione - Pagamento (alta)": (
        "Pagamento duplicato fattura fornitore",
        "Risulta un pagamento duplicato su fattura del mese. Chiedo verifica e rimborso, è urgente per chiusura contabile."
    ),
    "Amministrazione - Sollecito (media)": (
        "Sollecito invio fattura ricambi",
        "Non trovo la fattura relativa ai ricambi ordinati. Potete reinviarla? Grazie."
    ),
    "Commerciale - Stato ordine (bassa)": (
        "Richiesta stato ordine ricambi",
        "Vorrei aggiornamento su stato ordine e tempi di consegna previsti per i ricambi richiesti."
    ),
    "Commerciale - Offerta (media)": (
        "Richiesta preventivo flotta aziendale",
        "Richiedo un preventivo aggiornato per fornitura flotta e condizioni commerciali, con indicazione tempi."
    ),
}

# Stato persistente dei campi
if "title" not in st.session_state:
    st.session_state["title"] = ""
if "body" not in st.session_state:
    st.session_state["body"] = ""

scelta = st.selectbox("Esempi rapidi (per demo)", list(ESEMPI.keys()))

colA, colB, colC = st.columns(3)
with colA:
    if st.button("Carica esempio"):
        st.session_state["title"], st.session_state["body"] = ESEMPI[scelta]
        st.rerun()

with colB:
    if st.button("Esempio casuale"):
        k = random.choice([k for k in ESEMPI.keys() if k != "Seleziona un esempio..."])
        st.session_state["title"], st.session_state["body"] = ESEMPI[k]
        st.rerun()

with colC:
    if st.button("Pulisci campi"):
        st.session_state["title"] = ""
        st.session_state["body"] = ""
        st.rerun()

title = st.text_input("Titolo (oggetto)", key="title")
body = st.text_area("Descrizione", key="body", height=120)

if st.button("Predici"):
    text = combine_text(title, body)

    pred_cat = cat_model.predict([text])[0]
    pred_pri = pri_model.predict([text])[0]
    # Confidenza (se il modello supporta predict_proba)
    def get_confidence(model, x_text):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([x_text])[0]
            best_idx = int(proba.argmax())
            best_label = model.classes_[best_idx]
            best_score = float(proba[best_idx])
            return best_label, best_score, proba
        return None, None, None

    cat_label, cat_conf, _ = get_confidence(cat_model, text)
    pri_label, pri_conf, _ = get_confidence(pri_model, text)

    st.success(f"Categoria prevista: **{pred_cat}**")
    st.warning(f"Priorità prevista: **{pred_pri}**")

    # aggiunto label della confidenza cotegoria 
    if cat_conf is not None:
        st.info(f"Confidenza categoria: **{cat_conf:.2f}**")
    else:
        st.info("Confidenza categoria: non disponibile per questo modello.")

    if pri_conf is not None:
        st.info(f"Confidenza priorità: **{pri_conf:.2f}**")
    else:
        st.info("Confidenza priorità: non disponibile per questo modello.")

    # 5 parole influenti per categoria
    _, top_words_cat = top_influential_words(text, cat_model, top_k=5)
    st.write("**Top 5 parole influenti (categoria):**")
    if top_words_cat:
        st.table(pd.DataFrame(top_words_cat, columns=["feature", "score"]))
    else:
        st.write("Nessuna feature influente rilevabile (testo troppo generico).")

    # 5 parole influenti per priorità
    _, top_words_pri = top_influential_words(text, pri_model, top_k=5)
    st.write("**Top 5 parole influenti (priorità):**")
    if top_words_pri:
        st.table(pd.DataFrame(top_words_pri, columns=["feature", "score"]))
    else:
        st.write("Nessuna feature influente rilevabile (testo troppo generico).")

st.divider()

st.subheader("2) Predizione batch (CSV)")
st.write("Carica un CSV con colonne: **id,title,body**. Scarichi un CSV con **pred_category** e **pred_priority**.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in ["id", "title", "body"] if c not in df.columns]
    if missing:
        st.error(f"Mancano colonne: {missing}")
    else:
        X = [combine_text(t, b) for t, b in zip(df["title"], df["body"])]
        df["pred_category"] = cat_model.predict(X)
        df["pred_priority"] = pri_model.predict(X)

        st.write("Anteprima risultati:")
        st.dataframe(df.head(10))

        out_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Scarica CSV con predizioni",
            data=out_csv,
            file_name="ticket_predictions.csv",
            mime="text/csv"
        )
from pathlib import Path

st.divider()
st.subheader("3) Grafici e report (output)")

PLOT_FILES = [
    ("Distribuzione categorie", "category_distribution.png"),
    ("Distribuzione priorità", "priority_distribution.png"),
    ("Confusion matrix categoria", "category_confusion_matrix.png"),
    ("Confusion matrix priorità", "priority_confusion_matrix.png"),
]

for title_plot, fname in PLOT_FILES:
    p = Path(fname)
    if p.exists():
        st.write(f"**{title_plot}**")
        st.image(str(p), width="stretch")

        #st.image(str(p), use_container_width=True) deprecata dal 31/12/2025 
    else:
        st.caption(f"File non trovato: {fname} (genera i grafici con lo script di training).")
