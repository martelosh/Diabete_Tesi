import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Divide il dataset in training e test set."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_models_cross_validation(x_train, y_train, n_splits=5, random_state=42):
    """
    Esegue la cross-validazione su una serie di modelli di classificazione
    e stampa l'accuratezza media per ciascuno.

    Args:
        x_train (array-like): Le feature del dataset di training.
        y_train (array-like): Le etichette target del dataset di training.
        n_splits (int, optional): Il numero di split da utilizzare per la K-Fold cross-validation.
                                  Default è 5.
        random_state (int, optional): Il seed per il generatore di numeri casuali
                                      utilizzato in KFold per la riproducibilità.
                                      Default è 42.

    Returns:
        dict: Un dizionario contenente l'accuratezza media per ciascun modello.
    """

    logistic_regression_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=random_state)
    sgd_classifier_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    decision_tree_model = DecisionTreeClassifier(random_state=random_state)
    random_forest_model = RandomForestClassifier(random_state=random_state)
    knn_model = KNeighborsClassifier()
    xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    lightgbm_model = LGBMClassifier(random_state=random_state)

    models_to_evaluate = {
        "LogisticRegression": logistic_regression_model,
        "SGDClassifier": sgd_classifier_model,
        "DecisionTree": decision_tree_model,
        "RandomForest": random_forest_model,
        "KNN": knn_model,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model
    }

    # Inizializzazione del KFold
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    print("Inizio valutazione dei modelli con K-Fold Cross-Validation (accuratezza):")
    print("-" * 70)

    for name, model_instance in models_to_evaluate.items():
        print(f"Valutazione di: {name}...")
        scores = cross_val_score(model_instance, x_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1) # n_jobs=-1 usa tutti i core disponibili
        mean_accuracy = scores.mean()
        results[name] = mean_accuracy
        print(f"  {name} = {mean_accuracy:.4f}") # Formattazione per 4 cifre decimali
        print("-" * 70)

    return results