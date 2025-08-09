import os
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV

param_grids = {
    "SGDClassifier": {
        "loss": ["hinge", "log_loss"],
        "penalty": ["l2", "l1"],
        "alpha": [0.0001, 0.001],
    },
    "DecisionTree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    "XGBoost": {
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
    },
    "LightGBM": {
        "num_leaves": [31, 50],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
    }
}

def run_grid_search_and_save(estimator, param_grid, x_train, y_train, model_name,
                             cv=5, scoring='accuracy', n_jobs=-1, verbose=1):

    folder_path = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
    os.makedirs(folder_path, exist_ok=True)

    model_file = os.path.join(folder_path, f"{model_name}_optimized_model.pkl")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(x_train, y_train)

    best_score = float(grid_search.best_score_)  # accuracy media CV
    _best_to_meta(model_name=model_name, score=best_score)

    best_params = grid_search.best_params_
    print(f"Migliori parametri trovati: {best_params}")

    best_model = estimator.__class__(**best_params)
    best_model.fit(x_train, y_train)

    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Modello salvato in {model_file}")

    return best_model, grid_search

import json
from pathlib import Path

def _best_to_meta(model_name: str, score: float):
    """Aggiorna data/grid_search_results/model_meta.json con info sklearn."""
    artifacts = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
    artifacts.mkdir(parents=True, exist_ok=True)
    meta_path = artifacts / "model_meta.json"

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["sklearn_model"] = model_name
    meta["sklearn_score"] = score
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")