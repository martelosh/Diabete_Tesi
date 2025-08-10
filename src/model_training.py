import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import ClassifierMixin

# Modelli base
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Divide in train/test mantenendo la distribuzione delle classi (stratify).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target '{target_column}' non trovata")
    X = df.drop(columns=[target_column])
    y = df[target_column].astype("int")
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # <-- stratificazione
    )


def _candidate_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    models: dict[str, ClassifierMixin] = {
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "LogReg": LogisticRegression(max_iter=1000),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=random_state, verbosity=0, n_estimators=300
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=random_state, verbosity=-1, n_estimators=300)
    return models


def evaluate_models_cross_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Valuta i modelli in K-Fold (accuracy) e restituisce:
    - dict con accuracy media per modello,
    - best_estimator FITTATO su tutto il train,
    - best_model_name (stringa).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: dict[str, float] = {}
    best_name = None
    best_score = -np.inf

    for name, model in _candidate_models(random_state).items():
        scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)
        mean_acc = float(np.mean(scores))
        results[name] = mean_acc
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name

    assert best_name is not None
    best_estimator = _candidate_models(random_state)[best_name]
    best_estimator.fit(x_train, y_train)  # fit finale sul train completo

    return results, best_estimator, best_name


def evaluate_on_test(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Ritorna accuracy, classification_report (testo) e confusion_matrix (lista di liste)."""
    y_pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "grid_search_results"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_keras(input_dim: int) -> Sequential:
    """MLP 3-classi (softmax). Si aspetta input già preprocessato."""
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))  # 3 classi fisse
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_keras_simple(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_val: pd.DataFrame, y_val: pd.Series,
    max_epochs: int = 50,
):
    """
    Allena un MLP 3-classi su dati GIÀ preprocessati (scalati).
    Restituisce (model, val_accuracy).
    """
    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np   = np.asarray(x_val,   dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int32)
    y_val_np   = np.asarray(y_val,   dtype=np.int32)

    model = _build_keras(input_dim=x_train_np.shape[1])

    ckpt_path = ARTIFACTS_DIR / "keras_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_accuracy", save_best_only=True, save_weights_only=False),
    ]

    model.fit(
        x_train_np, y_train_np,
        validation_data=(x_val_np, y_val_np),
        epochs=max_epochs, batch_size=64, verbose=0, callbacks=callbacks,
    )

    _, val_acc = model.evaluate(x_val_np, y_val_np, verbose=0)
    return model, float(val_acc)
