# Diabete
Progetto finale ITS

# Avvio applicazione web
Eseguire il seguente comando:
    streamlit run streamlit/main_streamlit.py


# Uso del retrain finale
Raccogli un po’ di record da Streamlit → data/training_feedback.csv.
Lancia:
    python -m src.retrain
Questo farà append al DB, retrainerà e aggiornerà i file in data/grid_search_results/.

# Metriche di andamento
Esegui quando vuoi aggiornare il report:
    python -m src.metrics
Il file data/metrics/weekly_report.csv conterrà:
week_start | tests | accuracy
