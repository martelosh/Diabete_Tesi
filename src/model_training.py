import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Divide il dataset in training e test set."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    y = y.astype('int')
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

    sgd_classifier_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    decision_tree_model = DecisionTreeClassifier(random_state=random_state)
    random_forest_model = RandomForestClassifier(random_state=random_state)
    knn_model = KNeighborsClassifier()
    xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, verbosity=0)
    lightgbm_model = LGBMClassifier(random_state=random_state, verbosity=-1)

    models_to_evaluate = {
        "SGDClassifier": sgd_classifier_model,
        "DecisionTree": decision_tree_model,
        "RandomForest": random_forest_model,
        "KNN": knn_model,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model
    }

    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}

    for name, model_instance in models_to_evaluate.items():
        print(f"Valutazione di: {name}...")
        scores = cross_val_score(model_instance, x_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1)
        mean_accuracy = scores.mean()
        results[name] = mean_accuracy
    
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    best_estimator = models_to_evaluate[best_model_name]
    return {best_model_name: float(best_accuracy)}, best_estimator, best_model_name





def tune_keras_model(x_train, y_train, x_val, y_val, max_epochs=50):
    """Esegue tuning e training di un modello Keras e restituisce il migliore e la sua accuratezza."""

    def build_model(hp):
        model = Sequential()
        model.add(Dense(
            units=hp.Int('units_input', min_value=64, max_value=256, step=32),
            activation='relu',
            input_shape=(x_train.shape[1],)
        ))
        model.add(Dropout(hp.Float('dropout_input', 0.1, 0.5, step=0.1)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                activation='relu'
            ))
            model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))

        model.add(Dense(len(set(y_train)), activation='softmax'))

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train,
                 validation_data=(x_val, y_val),
                 epochs=max_epochs,
                 batch_size=64,
                 callbacks=[stop_early],
                 verbose=0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=max_epochs,
                        batch_size=64,
                        callbacks=[stop_early],
                        verbose=0)

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    return model, val_accuracy
