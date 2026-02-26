# src/generate_dataset.py
import random
import pandas as pd
from utils import clean_text

CATEGORIES = ["Amministrazione", "Tecnico", "Commerciale"]
PRIORITIES = ["bassa", "media", "alta"]

PRIORITY_TRIGGERS = {
    "alta": [
        "urgente", "bloccante", "servizio fermo", "down", "errore 500",
        "crash", "timeout", "scadenza oggi", "pagamento duplicato", "rimborso"
    ],
    "media": [
        "non funziona", "lento", "problema", "sollecito", "entro domani",
        "conferma pagamento", "accesso", "login"
    ]
}

def assign_priority(title: str, body: str) -> str:
    """
    Etichettatura priorità basata su keyword (richiesta dalla traccia).
    Prima alta, poi media, altrimenti bassa.
    """
    text = clean_text(f"{title} {body}")
    for trig in PRIORITY_TRIGGERS["alta"]:
        if trig in text:
            return "alta"
    for trig in PRIORITY_TRIGGERS["media"]:
        if trig in text:
            return "media"
    return "bassa"

def generate_synthetic_dataset(n_total: int = 500, seed: int = 7, out_csv: str = "data/tickets_sintetici.csv"):
    random.seed(seed)

    # Lessico tipico per categoria (title + body)
    admin_titles = [
        "Fattura non ricevuta", "Pagamento duplicato", "Errore importo fattura",
        "Scadenza fattura oggi", "Richiesta nota di credito", "IBAN per bonifico",
        "Sollecito pagamento", "Pagamento non registrato"
    ]
    admin_bodies = [
        "Non trovo la fattura del mese, potete reinviarla?",
        "Ho pagato due volte, chiedo rimborso urgente.",
        "L'importo non corrisponde al contratto, serve verifica.",
        "Mi serve conferma pagamento entro domani.",
        "Potete emettere nota di credito per fattura errata?",
        "Indicatemi l'IBAN corretto per il bonifico.",
        "Risulta insoluto ma ho ricevuta, pagamento non registrato."
    ]

    tech_titles = [
        "Errore 500 in dashboard", "App non si avvia", "Crash dopo aggiornamento",
        "Login non funziona", "Sistema lento", "Timeout durante upload",
        "API restituisce 401", "Servizio fermo"
    ]
    tech_bodies = [
        "Da stamattina errore 500 quando apro la schermata principale.",
        "L'app si chiude subito, comportamento bloccante.",
        "Dopo aggiornamento crash frequente, è urgente.",
        "Non riesco ad accedere, login non funziona con credenziali corrette.",
        "Il sistema è lento e spesso non risponde.",
        "Upload va in timeout e non completa l'operazione.",
        "Integrazione API fallisce con 401, non posso lavorare.",
        "Servizio fermo, sistema down, priorità alta."
    ]

    sales_titles = [
        "Richiesta preventivo", "Info offerta Pro", "Stato ordine",
        "Tempi di consegna", "Sconto per quantità", "Upgrade piano",
        "Cambio ordine", "Disponibilità prodotto"
    ]
    sales_bodies = [
        "Vorrei un preventivo per 20 licenze e assistenza.",
        "Quali differenze tra piano Base e Pro?",
        "Qual è lo stato del mio ordine e quando arriva?",
        "Mi servono tempi di consegna stimati.",
        "Possiamo avere sconto per quantità su ordine ricorrente?",
        "Vorrei fare upgrade del piano dal prossimo mese.",
        "Posso modificare l'ordine già inserito?",
        "Disponibilità prodotto e costi di spedizione."
    ]

    # “Variazioni” per rendere il dataset meno ripetitivo
    spices = [
        "", " per favore", " grazie", " il prima possibile", " entro oggi",
        " entro domani", " non urgente", " è urgente", " richiesta bloccante"
    ]
    ids = [f"ID#{i}" for i in range(1000, 9999)]

    rows = []
    # Bilanciamento: distribuisco per categoria in modo uniforme
    per_cat = n_total // 3
    remainder = n_total - per_cat * 3
    counts = {"Amministrazione": per_cat, "Tecnico": per_cat, "Commerciale": per_cat}
    for c in list(counts.keys())[:remainder]:
        counts[c] += 1

    current_id = 1
    for cat, n in counts.items():
        for _ in range(n):
            if cat == "Amministrazione":
                title = random.choice(admin_titles)
                body = random.choice(admin_bodies)
            elif cat == "Tecnico":
                title = random.choice(tech_titles)
                body = random.choice(tech_bodies)
            else:
                title = random.choice(sales_titles)
                body = random.choice(sales_bodies)

            # Inserisco ID finto + spice
            title2 = f"{title} {random.choice(ids)}"
            body2 = f"{body}{random.choice(spices)}."

            priority = assign_priority(title2, body2)

            rows.append({
                "id": current_id,
                "title": title2,
                "body": body2,
                "category": cat,
                "priority": priority
            })
            current_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Creato dataset: {out_csv} | righe={len(df)}")
    print(df.head(5))

if __name__ == "__main__":
    generate_synthetic_dataset(n_total=500, seed=7, out_csv="data/tickets_sintetici.csv")

