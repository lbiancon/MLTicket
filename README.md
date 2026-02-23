# MLTicket
Repository that contain the project MLTichek for Pegaso PW by Bianconi Luca 0312301652
# Triage automatico dei ticket con Machine Learning (Prototipo minimale)

## descrizione degli step eseguiti
```bash
Step 1 — struttura progetto
"Inizializzazione struttura progetto MLTicket"
Contiene:
cartelle src/, data/, models/

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

Include:
- generazione dataset sintetico
- training + valutazione
- dashboard Streamlit (singolo ticket + batch CSV)

## Requisiti
- Python 3.10+ (consigliato 3.11)
- pip

## Installazione
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

## Esecuzione (ordine corretto)
# installa le dipendenze 
pip install -r requirements.txt

## Genera dataset sintetico:
python src/generate_dataset.py

## Addestra e valuta i modelli (genera anche PNG e salva i modelli in /models):
python src/train_eval.py

## Avvia la dashboard:
python -m streamlit run src/app.py

## Batch prediction
## Carica un CSV con colonne:
## id,title,body

## La dashboard produce un CSV con:
pred_category, pred_priority

## per eseguirlo senza lanciare comando

dasto destro su il file  `run_all.ps1` (Windows) nella root del progetto e poi esegui con powershell


