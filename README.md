# MLTicket
Repository that contain the project MLTichek for Pegaso PW by Bianconi Luca 0312301652
# Triage automatico dei ticket con Machine Learning (Prototipo minimale)

## descrizione degli step eseguiti
```bash
Step 1 — struttura progetto
"Inizializzazione struttura progetto MLTicket"
Contiene:
cartelle src/, data/, models/, reports/

## README iniziale
file vuoti 

## Step 2 — generatore dataset
"Aggiunto generatore dataset sintetico ticket"
Contiene:
script generazione CSV
prime regole per priorità
output data/tickets_sintetici.csv

## Step 3 — preprocessing testo
"Implementato preprocessing testuale base"
Contiene:
pulizia testo
funzione combine_text
tokenizzazione base

## Step 4 — modello categoria
"Prima pipeline di classificazione categoria ticket"
Contiene:
-TF-IDF
-Logistic Regression
-training categoria

## Step 5 — modello priorità
"Aggiunto modello di stima priorità ticket"
Contiene:
-secondo modello
-training priorità

## Step 6 — valutazione modelli
"Valutazione modelli con accuracy, F1 e confusion matrix"
Contiene:
-classification_report
-confusion matrix
-salvataggio PNG

## Step 7 — salvataggio modelli
"Salvataggio modelli addestrati in formato joblib"
Contiene:
-models/*.joblib

## Step 8 — dashboard base
"Dashboard Streamlit per predizione singolo ticket"
Contiene:
-app.py con input e predizione

## Step 9 — spiegabilità
"Visualizzazione parole influenti per predizione"
Contiene:
-top_influential_words
-tabelle feature-score

## Step 10 — batch CSV
"Predizione batch su CSV con esportazione risultati"

## Step 11 — UX dashboard
"Aggiunti esempi precompilati e generazione casuale ticket"

## Step 12 — confidenza
 "Aggiunta visualizzazione confidenza predizioni"

## Step 13 — grafici in dashboard
"Visualizzazione grafici di valutazione in dashboard"

## Step 14 — pulizia finale
"Pulizia codice e aggiornamento README"

## Obiettivo
Classificare ticket testuali in:
- Categoria: Amministrazione / Tecnico / Commerciale
- Priorità: bassa / media / alta
e fornire:
- Valutazione semplice (train/test split 80/20, accuracy, F1 macro, confusion matrix)
- Dashboard per predizione singola e batch + 5 parole più influenti

## Setup
```bash
pip install -r requirements.txt
mkdir -p data models
python src/generate_dataset.py
python src/train_eval.py
streamlit run src/app.py

