# 🔎 Valutazione Rischio Diabete

**Autori:** Dicorato Martina, Kirollos Seif  
**Versione:** 1.0  

> ⚠️ **Nota medica**: questo strumento ha scopi **informativi/educativi** e **non sostituisce un consulto medico**.

---

## ✨ Cosa fa

- Modello ML per classificare il rischio diabete (classi `0/1/2`).
- Due interfacce **Streamlit**:
  - **Test (`streamlit_test/`)**: flusso completo con salvataggio feedback e monitoraggio.
  - **Produzione (`streamlit_prod/`)**: UI curata, mostra probabilità della classe, contatti ospedali per comune (mappa), log interazioni per analisi.
- Dati e report salvati in `data/` (CSV + metriche).
- Notebook per analisi rapida dei dati di produzione.

---

## 🗂️ Struttura della repository

.
├─ data/
│ ├─ diabete_data.csv # dataset di origine
│ ├─ ospedali_milano_comuni_mapping.csv # mappa COMUNE → ospedale/contatti (anche lat/lon)
│ ├─ feedback_test.csv # feedback app Test (si crea/scrive dalla app)
│ ├─ prod_interactions.csv # log interazioni app Prod (si crea/scrive dalla app)
│ ├─ grid_search_results/
│ │ ├─ LightGBM_optimized_model.pkl # miglior modello sklearn (se presente)
│ │ ├─ best_keras_model.(keras|h5) # miglior modello keras (se presente)
│ │ ├─ scaler.pkl # scaler per inference (se presente)
│ │ └─ model_meta.json # meta: feature_order, punteggi, ecc.
│ └─ metrics/ # report generati dalle app/notebook
│ ├─ weekly_report.csv
│ ├─ by_model_report.csv
│ └─ confusion_matrix_overall.csv
│
├─ notebooks/
│ ├─ data_analysis.ipynb
│ └─ prod_interactions_analysis.ipynb # EDA semplice sul log di produzione
│
├─ src/
│ ├─ init.py
│ ├─ data_preprocessing.py
│ ├─ grid_search.py
│ ├─ main.py
│ ├─ model_training.py
│ ├─ refresh_db_table.py
│ ├─ utils.py
│ └─ from_streamlit/metrics_report.py
│
├─ streamlit_test/
│ └─ main_streamlit_test.py
│
├─ streamlit_prod/
│ └─ main_streamlit_prod.py
│
├─ requirements.txt / pyproject.toml / uv.lock
├─ .python-version (se presente)
└─ README.md


---

## 🧰 Requisiti

- **Python** 3.10+ (consigliato 3.11/3.12)  
- Librerie da `requirements.txt` (o `pyproject.toml`)  
- (Opzionale) **Git** configurato per commit automatici dei CSV dalla app di test  

### Installazione

```bash
# creare e attivare un venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# installare i pacchetti
pip install -r requirements.txt
▶️ Avviare le app
1) Ambiente TEST
Flusso: Home → Form → Monitoraggio

Salva i feedback in data/feedback_test.csv (se non esiste, la app lo crea).

Genera report in data/metrics/.

bash
Copy
Edit
streamlit run streamlit_test/main_streamlit_test.py
📌 Note:

La pagina Monitoraggio rigenera i report se mancano.

Supporto opzionale al commit automatico su GitHub dei CSV (vedi sezione Git auto-sync).

2) Ambiente PRODUZIONE
UI più curata, mostra classe + probabilità.

Ricerca contatti ospedalieri per comune (data/ospedali_milano_comuni_mapping.csv) con mappa.

Log interazioni in data/prod_interactions.csv (crea e appende automaticamente).

bash
Copy
Edit
streamlit run streamlit_prod/main_streamlit_prod.py
📌 Suggerimenti:

Verifica che data/ospedali_milano_comuni_mapping.csv contenga le colonne attese:

Copy
Edit
Comune, Ospedale di riferimento, Indirizzo Ospedale, Telefono,
Prenotazioni/CUP, Note, lat, lon
La mappa si centra su Milano; quando selezioni un comune, lo zoom si aggiorna.

🧠 Modelli & Pipeline
Gestione modello migliore → src/utils.py:

legge data/grid_search_results/model_meta.json (feature_order, punteggi, tipo modello).

carica LightGBM_optimized_model.pkl (sklearn) o best_keras_model.(keras|h5) (Keras) + scaler.

inferenza coerente con preprocess_for_inference(...).

Training pipeline:

src/grid_search.py → ricerca modelli e salvataggio artefatti.

src/model_training.py → funzioni di training.

src/main.py → pipeline end-to-end.

🧾 Dati salvati dalle app
Test → data/feedback_test.csv
Esempio colonne:

sql
Copy
Edit
HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income,
Predicted, Diabetes_012, timestamp, model_type, model_artifact
Produzione → data/prod_interactions.csv
Esempio colonne:

sql
Copy
Edit
timestamp, session_id, event_type, HighBP, HighChol, CholCheck, BMI, Smoker,
Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
Education, Income, predicted_class, probability, comune, ospedale, telefono,
indirizzo, prenotazioni, note
Report (monitor) → data/metrics/
weekly_report.csv

by_model_report.csv

confusion_matrix_overall.csv

📒 Notebook
notebooks/prod_interactions_quick_eda.ipynb
EDA rapida con:

conteggi eventi per tipo,

trend giornalieri,

torta classi predette,

top comuni,

tempo prediction → primo contact_view,

export CSV sintesi in data/metrics/.

📌 Nei notebook __file__ non esiste → incluso un helper che risale le cartelle fino a trovare data/ e src/.

☁️ DB Cloud (opzionale)
Supporto a DB remoto (PostgreSQL/MySQL) via src/refresh_db_table.py.

Non obbligatorio → per semplicità, attualmente si usano CSV locali.

🔄 Git auto-sync (opzionale)
Nella app Test, utility che tenta:

bash
Copy
Edit
git add <file>
git commit -m "<msg>"
git pull --rebase origin main
git push origin main
Per funzionare:

Git deve essere installato.

Remote origin → GitHub configurato.

Credenziali/token disponibili.

Se non configurato, i CSV vengono comunque salvati in locale (UI mostra un toast di warning).

🧩 Convenzioni & scelte progettuali
Path robusti:

Script/Streamlit → PROJECT_ROOT = Path(__file__).resolve().parents[1].

Notebook → helper che risale fino a data/ e src/.

Separazione ambienti:

Test = sperimentazione + monitoraggio (feedback_test.csv).

Prod = UI pulita e centrata sull’utente (prod_interactions.csv).

Tracciabilità: CSV di test/prod permettono analisi utilizzo + qualità predizioni.

Ospedali/contatti: file unico data/ospedali_milano_comuni_mapping.csv.

UI design: pulsanti centrati, card, badge, mappe, popup raccomandazioni per classe/probabilità.

🐞 Troubleshooting veloce
CSV non aggiornato su GitHub → salvataggio locale ≠ push.
Usa auto-sync (se configurato) o git add/commit/push manuale.

Notebook non trova i file → lancia kernel da notebooks/ o root.
Oppure usa l’helper incluso.

Mappa contatti vuota → controlla data/ospedali_milano_comuni_mapping.csv e i nomi colonne.
Per zoom su comune, includi lat/lon.