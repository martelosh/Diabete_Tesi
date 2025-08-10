# Progetto: Valutazione del Rischio Diabete – Documentazione

## Introduzione
Questo progetto fornisce due applicazioni Streamlit (demo/test e produzione) che stimano il rischio di diabete in tre classi (0 = nessun diabete, 1 = pre-diabete, 2 = diabete) a partire da variabili cliniche e di stile di vita.  
È pensato per supportare operatori sanitari e medici: l’utente compila un breve form, riceve la classe stimata (e in produzione anche la probabilità), può lasciare un **feedback** (nella versione test), e il sistema genera **metriche** settimanali per monitorare l’andamento. Il flusso supporta **retraining** periodico del modello.

## Architettura (panoramica rapida)
1. **Origine dati**: tabella `diabetes_data` su DB (MySQL) oppure CSV locali in `data/`.
2. **Training script** (`src/main.py`): 
   - carica i dati puliti/scalati dal DB;
   - fa split stratificato;
   - valuta più modelli classical ML via cross-validation (seleziona il migliore);
   - lancia grid search sui parametri del modello vincente e salva il modello ottimizzato in `data/grid_search_results`;
   - allena anche un MLP Keras su validation (opzionale) e salva il miglior checkpoint;
   - aggiorna `model_meta.json` con i punteggi medi.
3. **Utils**: caricamento del “miglior” modello (Sklearn vs Keras) in base alle metriche salvate; normalizzazione dell’ordine delle feature in inference; predizione.
4. **Streamlit**:
   - **Test/demo** (`streamlit_test/main_streamlit_test.py`): form + salvataggio feedback + monitoraggio; genera i CSV delle metriche (weekly/by_model/confusion) leggendo `data/training_feedback.csv`.
   - **Produzione** (`streamlit_prod/main_streamlit_prod.py`): form senza feedback; mostra classe + probabilità; chat “leggera” per proporre contatti utili filtrando un CSV di reference locali (nessun modello linguistico esterno).
5. **Metriche e Report** (`src/from_streamlit/metrics_report.py`): funzioni riutilizzabili per costruire i report a partire dai CSV creati dalle app.
6. **Artefatti** (`data/grid_search_results`): modelli salvati (.pkl per Sklearn; .h5/.keras per Keras), `model_meta.json` con punteggi e ordine delle feature, e altri file ausiliari.

---

## Struttura della repository

```
.
├─ data/
│  ├─ diabete_data.csv                       # dataset di partenza (se usato localmente)
│  ├─ training_feedback.csv                  # feedback raccolti dalla demo/test
│  ├─ ospedali_milano_comuni_mapping.csv     # rubrica contatti sanità locale (per chat in prod)
│  ├─ metrics/
│  │  ├─ weekly_report.csv
│  │  ├─ by_model_report.csv
│  │  └─ confusion_matrix_overall.csv
│  └─ grid_search_results/
│     ├─ LightGBM_optimized_model.pkl        # esempio: modello Sklearn selezionato dalla grid search
│     ├─ best_keras_model.h5 / .keras        # checkpoint Keras migliore su validation
│     └─ model_meta.json                     # punteggi e info per selezione automatica del modello
│
├─ src/
│  ├─ main.py                                # orchestratore training + grid search + (opz.) Keras
│  ├─ data_preprocessing.py                  # funzioni DB/csv + pulizia + scaling (per refresh DB)
│  ├─ model_training.py                      # split stratificato, CV modelli, test, e Keras training
│  ├─ grid_search.py                         # griglie iperparametri e grid search con salvataggio
│  ├─ utils.py                               # load_best_model, preprocess_for_inference, predict
│  └─ from_streamlit/
│     └─ metrics_report.py                   # build_weekly_feedback_report / build_usage_report
│
├─ streamlit_test/
│  └─ main_streamlit_test.py                 # app demo: form + feedback + monitor (router interno)
│
├─ streamlit_prod/
│  ├─ main_prod.py                           # home + chat contatti + routing verso pages/1_Form.py
│  └─ pages/
│     └─ 1_Form.py                           # form produzione (classe + probabilità)
│
├─ .env                                      # credenziali DB (SQL_USERNAME, SQL_PASSWORD, ...)
└─ requirements.txt                          # versioni librerie (consigliato allineare Sklearn)
```

---

## Dettaglio file per file

### `src/main.py`
- Legge `.env` per le credenziali SQL.
- Con `create_db_engine` + `test_connection` verifica l’accesso al DB.
- Carica la tabella **già pulita e scalata** (`diabetes_data`) dal DB.
- Esegue `split_data` (stratificato) per ottenere train/test.
- Esegue `evaluate_models_cross_validation` su 5–fold CV per vari modelli (SGD, DT, RF, KNN, LogReg, XGB, LGBM).
- Sceglie il migliore per accuracy media CV e lancia **grid search** con `run_grid_search_and_save` (salva il miglior modello `.pkl` e aggiorna `model_meta.json`).
- Allena **Keras MLP** con `train_keras_simple` (train/val split interno) e aggiorna `model_meta.json` con `keras_score`.
- Output: artefatti in `data/grid_search_results` + print dei risultati.

### `src/model_training.py`
- `split_data(df, target_column, ...)`: train/test con **stratify=y**.
- `_candidate_models()`: definisce il set di modelli base (inclusi XGBoost/LightGBM se installati).
- `evaluate_models_cross_validation(x_train, y_train, ...)`: KFold CV, restituisce le accuracy medie, il **best_estimator** fit su full-train e il **nome** del modello migliore.
- `evaluate_on_test(model, x_test, y_test)`: ritorna accuracy, classification report e confusion matrix (per uso diagnostico).
- `train_keras_simple(...)`: MLP 3-classi (softmax). Input **già preprocessato** (niente scaler qui). EarlyStopping + ModelCheckpoint, ritorna modello Keras e val_accuracy.

### `src/grid_search.py`
- `param_grids`: dizionario con iperparametri per ciascun modello classico (SGD, DT, RF, KNN, XGB, LGBM, ecc.).
- `run_grid_search_and_save(estimator, param_grid, x_train, y_train, model_name, ...)`:
  - esegue `GridSearchCV` con `scoring="accuracy"`;
  - **fit** del modello con i best params su tutto il train;
  - salvataggio del modello `.pkl` in `data/grid_search_results/{model_name}_optimized_model.pkl`;
  - aggiorna `model_meta.json` con il punteggio `sklearn_score` del migliore (CV mean).  
  - ritorna `(best_model, grid_search_obj)`.

### `src/utils.py`
- `load_best_model()`:
  - apre `data/grid_search_results/model_meta.json` e confronta `sklearn_score` vs `keras_score`;
  - carica il modello Sklearn (`*_optimized_model.pkl`) o Keras (`best_keras_model.h5/.keras`) in base al punteggio;
  - ritorna `(model, "sklearn"|"keras", meta)`.
- `preprocess_for_inference(df_row, meta)`:
  - **non** aplica scaler (i dati sono già puliti/scalati lato DB);
  - allinea l’ordine delle colonne a `meta["feature_order"]` se presente.
- `predict_with_model(model, model_type, X)`:
  - Sklearn: `predict` e cast a `int`;
  - Keras: `predict` e `argmax` sulla softmax.

### `src/data_preprocessing.py`
- `create_db_engine`, `test_connection`: utilità SQLAlchemy per MySQL (driver `pymysql`).
- `load_data_from_csv`, `preprocess_data`: funzioni per pipeline di pulizia/scaling (usate quando si ricarica il DB da CSV “grezzo”).
- `append_dataframe_to_db`, `import_dataframe_to_db`: scrittura su tabella (append/replace).  
  > Nota: la **pulizia/scaling** viene fatta **a monte** nello script di refresh del DB, così in runtime non si scala più.

### `src/from_streamlit/metrics_report.py`
- `build_weekly_feedback_report(feedback_csv: PathLike)`:
  - legge `training_feedback.csv` (o path passato);
  - calcola `is_correct` (match `Predicted` vs `Diabetes_012`);
  - genera:
    - `metrics/weekly_report.csv` (tests per settimana + accuracy media);
    - `metrics/by_model_report.csv` (tests e accuracy per modello caricato);
    - `metrics/confusion_matrix_overall.csv` (matrice complessiva).
- `build_usage_report(usage_csv: PathLike)`:
  - stesso schema ma per un CSV di utilizzo **senza feedback** (se previsto).

### `streamlit_test/main_streamlit_test.py`
- App **single-file** con router (Home/Form/Monitor):
  - **Home**: mostra info e punteggi del modello caricato (dalle meta).
  - **Form**: input utente, calcolo predizione, **feedback** (sì/no + label corretta), salvataggio in `data/training_feedback.csv`.
  - **Monitor**: genera/aggiorna report settimanali e li mostra (chart + tabelle).  
    Se presente, usa `build_weekly_feedback_report` dal modulo; altrimenti un fallback inline.
- Nasconde la sidebar e cura l’estetica base (CSS inline).

### `streamlit_prod/main_prod.py` + `streamlit_prod/pages/1_Form.py`
- **Produzione**:
  - `main_prod.py`: Home + bottone per aprire `pages/1_Form.py`. Include **chat flottante**: quando c’è una predizione, mostra una “vignetta” e rende possibile cercare contatti locali. La funzione `get_nearby_contacts()` filtra **`data/ospedali_milano_comuni_mapping.csv`** per il comune inserito dall’utente. **Nessuna API esterna**.
  - `pages/1_Form.py`: Form di valutazione (senza feedback). Mostra **classe + probabilità**. Aggiorna lo stato condiviso per alimentare la chat sulla home.
- UI più curata (hero, pulsanti, pannello chat).

### `data/` e artefatti
- `training_feedback.csv`: generato dalla demo, contiene le righe con record utente, `Predicted`, `Diabetes_012` (feedback), `timestamp`, `model_type`, `model_artifact`.
- `metrics/`: tre CSV aggiornati dal modulo report o dal fallback inline.
- `grid_search_results/`: modelli e meta usati in inference; se `keras_score` > `sklearn_score`, **utils** carica Keras automaticamente.

### `.env`
- Variabili: `SQL_USERNAME`, `SQL_PASSWORD`, `SQL_HOST`, `SQL_PORT`, `SQL_DATABASE` (e opzionalmente `DB_TABLE`).  
- Esempio:
  ```env
  SQL_USERNAME=user
  SQL_PASSWORD=pass
  SQL_HOST=localhost
  SQL_PORT=3306
  SQL_DATABASE=diabetesdb
  DB_TABLE=diabetes_data
  ```

---

## Come eseguire

### 1) Training / Grid search
```bash
python -m src.main
```
Risultato: salva il miglior modello Sklearn e (se Keras riesce) anche il modello Keras, aggiornando `model_meta.json` per la **selezione automatica**.

### 2) Demo/Test (form + feedback + monitor)
```bash
streamlit run streamlit_test/main_streamlit_test.py --server.port 8501
```
- Compila form → calcola predizione → salva feedback → apri Monitor (genera & visualizza report).

### 3) Produzione (classe + probabilità + chat contatti)
```bash
streamlit run streamlit_prod/main_prod.py --server.port 8502
```
- Compila form (pages/1_Form.py) → classe + probabilità → chat “no-LLM” con filtro contatti.

### 4) Refresh tabella DB (opzionale)
- Script (esempio) che svuota e ricarica la tabella dal CSV **dopo** pulizia/scaling: `scripts/refresh_db_table.py` (se presente).  
- Scopo: avere dati già puliti/scalati in DB; a runtime e in Streamlit **non** si scala più.

---

## Note operative & Troubleshooting
- **Sklearn pickle warning**: se vedi *InconsistentVersionWarning* durante il load di un modello picklato, allinea la versione di scikit-learn usata in training con quella in produzione (o rilancia il training).
- **Mancanza feedback**: il monitoraggio richiede almeno 1 riga in `data/training_feedback.csv`.
- **Selezione automatica modello**: `utils.load_best_model()` sceglie tra Keras e Sklearn usando i punteggi salvati; se manca uno dei due, fa fallback all’altro.
- **Ordine delle feature**: in inference viene rispettato `meta["feature_order"]` se presente; assicurati che le colonne di input abbiano gli stessi nomi del training.

---

## Evoluzioni possibili
- Bilanciamento classi (pesi o resampling) se necessario.
- Aggiunta di spiegazioni locali (es. SHAP) per trasparenza del modello.
- Salvataggio su DB anche dei feedback (attualmente CSV) per audit & reporting centralizzati.
- Scheduling del retraining e del refresh DB (cron o workflow CI/CD).