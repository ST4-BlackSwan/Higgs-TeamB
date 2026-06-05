from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  # Importation de l'Early Stopping
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    Class implementing a sequential neural network classifier with Keras.
    Architecture V2 (64->64->64->32) épurée (sans Dropout) avec Early Stopping.
    """

    def __init__(self, train_data):
        self.model = Sequential()

        # Nombre de variables d'entrée (features physiques)
        n_dim = train_data.shape[1]

        # --- ARCHITECTURE V2 ÉPURÉE (SANS DROPOUT) ---
        # 1. Couche d'entrée + Première couche cachée
        self.model.add(Dense(64, input_dim=n_dim, activation="swish"))

        # 2. Deuxième couche cachée
        self.model.add(Dense(64, activation="swish"))

        self.model.add(Dense(64, activation="swish"))


        # 3. Troisième couche cachée
        self.model.add(Dense(64, activation="swish"))

        # 4. Quatrième couche cachée (transition douce)
        self.model.add(Dense(32, activation="swish"))

        self.model.add(Dense(32, activation="swish"))


        # 5. Couche de sortie (Classification binaire : Signal=1 / Background=0)
        self.model.add(Dense(1, activation="sigmoid"))

        # Compilation avec un Learning Rate de 0.001 adapté au grand batch
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.002),
            metrics=["accuracy"],
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, validation_data=None, weights_train=None, weights_val=None):
        # Normalisation des données d'entraînement
        X_train_scaled = self.scaler.fit_transform(train_data)

        # Préparation des données de validation si fournies
        validation_data_for_keras = None
        monitor_metric = 'loss'  # Par défaut, on surveille la perte d'entraînement

        if validation_data is not None:
            X_val_raw, y_val_raw = validation_data
            X_val_scaled = self.scaler.transform(X_val_raw)
            validation_data_for_keras = (X_val_scaled, y_val_raw)
            
            if weights_val is not None:
                validation_data_for_keras = (X_val_scaled, y_val_raw, weights_val)
            
            monitor_metric = 'val_loss'  # Si on a de la validation, on surveille val_loss

        # --- FILET DE SÉCURITÉ (EARLY STOPPING) ---
        early_stopper = EarlyStopping(
            monitor=monitor_metric,    # Surveille la courbe de perte appropriée
            patience=15,               # Tolère 15 époques sans amélioration avant de couper
            restore_best_weights=True, # Recommence à la fin avec les poids de la MEILLEURE époque
            verbose=1                  # Affiche un message explicite à l'arrêt
        )

        # --- LANCEMENT DE L'ENTRAÎNEMENT ---
        history = self.model.fit(
            X_train_scaled,
            y_train,
            sample_weight=weights_train,
            validation_data=validation_data_for_keras,
            epochs=100,                # Monté à 100 sous la protection de l'Early Stopping
            batch_size=2048,
            verbose=2,
            callbacks=[early_stopper]  
        )
        return history

    def predict(self, test_data):
        # Application des paramètres de standardisation calculés sur le Train set
        test_data = self.scaler.transform(test_data)

        # Prédiction accélérée également par lot
        return (
            self.model.predict(test_data, batch_size=2048).flatten().ravel()
        )