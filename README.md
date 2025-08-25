# 🩺 Progetto Diabete — Prevenzione e Supporto con ML e Streamlit

## 📌 Descrizione
Questo progetto implementa una piattaforma per la **pre-valutazione del rischio diabete** basata su:
- Un **modello di Machine Learning** (classi 0/1/2 → rischio basso/medio/alto).
- Due applicazioni **Streamlit**:
  - **Produzione**: form di predizione, rubrica contatti/mappa strutture sanitarie, chatbot informativo.
  - **Test/Monitor**: raccolta feedback, reportistica e metriche.
- Un **database cloud (AlwaysData)** per centralizzare i log.
- Un **chatbot** basato su **OpenAI API** con supporto RAG opzionale.

⚠️ **Disclaimer**: lo strumento è solo di supporto e **non sostituisce la diagnosi medica**.

---

## 📂 Struttura della repository
repo/
├─ streamlit_prod/ # app di produzione
│ ├─ main_streamlit_prod.py # form + contatti + mappa + chatbot + logging
│ └─ chatbot.py # modulo chatbot (OpenAI API + RAG opzionale)
│
├─ streamlit_test/ # app di test/monitoraggio
│ └─ main_streamlit_test.py
│
├─ src/
│ ├─ main.py # orchestratore end-to-end (training → artefatti)
│ ├─ utils.py # funzioni di preprocess e inferenza
│ ├─ data_preprocessing.py # pipeline preparazione dati
│ ├─ model_training.py # addestramento e salvataggio modelli
│ ├─ grid_search.py # ricerca iperparametri
│ ├─ build_gold_dataset.py # costruzione dataset dai feedback validati
│ └─ from_streamlit/metrics_report.py # report metriche da feedback_test.csv
│
├─ data/
│ ├─ diabete_data.csv # dataset storico
│ ├─ ospedali_milano_comuni_mapping.csv # comuni → strutture sanitarie
│ ├─ prod_interactions.csv # log produzione
│ ├─ feedback_test.csv # feedback test
│ ├─ grid_search_results/ # modelli e meta info
│ ├─ metrics/ # report automatici
│ └─ gold/ # dataset finale dai feedback
│
├─ notebooks/
│ ├─ data_analysis.py # analisi iniziale dataset
│ └─ prod_interactions_analysis.py # analisi visiva log produzione
│
├─ requirements.txt # dipendenze
├─ .env # variabili API (non versionato)
└─ README.md # questo file

yaml
Copy
Edit

---

## ⚙️ Installazione

### 1. Clona la repo
```bash
git clone <repo_url>
cd repo
2. Crea ambiente virtuale

python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
3. Installa dipendenze

pip install -r requirements.txt
🚀 Utilizzo
1. Genera il modello (prima delle app)

python -m src.main
Artefatti salvati in data/grid_search_results/:

modello (.pkl o .keras)

model_meta.json (include ordine delle feature)

scaler.pkl se previsto

2. Avvia app di produzione

streamlit run streamlit_prod/main_streamlit_prod.py
Funzionalità:

Form di predizione (BMI calcolato automaticamente).

Esito rischio con messaggi prudenziali.

Rubrica strutture sanitarie con mappa.

Chatbot (OpenAI, opzionale).

Logging locale e su DB cloud (AlwaysData).

3. Avvia app di test/monitoraggio

streamlit run streamlit_test/main_streamlit_test.py
Funzionalità:

Predizione demo + raccolta feedback.

Reportistica (weekly_report.csv, by_model_report.csv, confusion_matrix_overall.csv).

🤖 Chatbot
Configurazione .env:


OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
Esempio test:


python streamlit_prod/chatbot.py --test "Cos'è il diabete di tipo 2?"
📊 Notebook disponibili
data_analysis.py → analisi esplorativa dataset.

prod_interactions_analysis.py → analisi visiva log di produzione.

☁️ Deploy gratuito (demo)
Possibile deploy su Render o Streamlit Cloud.

Comando start:


streamlit run streamlit_prod/main_streamlit_prod.py --server.port $PORT --server.address 0.0.0.0
Richiede artefatti già presenti in data/grid_search_results/.

🛠️ Troubleshooting
Mancano gli artefatti → eseguire python -m src.main.

Colonne disallineate → usare preprocess_for_inference() in src/utils.py.

Problemi Git → git pull --rebase e risoluzione conflitti.

Chatbot non risponde → verificare .env e crediti API.

📅 Roadmap
Miglioramento pagina Analytics (log produzione).

Active Learning basato su incertezza.

Dashboard BI collegata a DB cloud.

Estensione chatbot con citazioni e RAG migliorato.

📖 Licenza
MIT — uso libero per scopi educativi e di ricerca.