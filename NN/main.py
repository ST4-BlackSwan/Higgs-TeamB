from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout  # <-- RÉIMPORTATION DU DROPOUT
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  # Importation de l'Early Stopping
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    Class implementing a sequential neural network classifier with Keras.
    Architecture V2 (64->64->32->32) RECONSTRUITE avec Dropout pour briser l'overfitting.
    """

    def __init__(self, train_data):
        self.model = Sequential()

        # Nombre de variables d'entrée (features physiques)
        n_dim = train_data.shape[1]

        # --- ARCHITECTURE V2 SÉCURISÉE ---
        # 1. Couche d'entrée + Première couche cachée
        self.model.add(Dense(64, input_dim=n_dim, activation="swish"))
        self.model.add(Dropout(0.3))  # 30% de déconnexion pour forcer la généralisation

        # 2. Deuxième couche cachée
        self.model.add(Dense(64, activation="swish"))
        self.model.add(Dropout(0.2))

        # 3. Troisième couche cachée
        self.model.add(Dense(32, activation="swish"))
        self.model.add(Dropout(0.2))

        # 4. Quatrième couche cachée (transition douce avant la sortie)
        self.model.add(Dense(32, activation="swish"))

        # 5. Couche de sortie (Classification binaire)
        self.model.add(Dense(1, activation="sigmoid"))

        # Compilation avec un Learning Rate de 0.001 (plus stable pour éviter les sur-ajustements)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),  # Réduit de 0.002 à 0.001
            metrics=["accuracy"],
        )
        self.scaler = StandardScaler()

    def fit(
        self,
        train_data,
        y_train,
        validation_data=None,
        weights_train=None,
        weights_val=None,
    ):
        # Normalisation des données d'entraînement
        X_train_scaled = self.scaler.fit_transform(train_data)

        # Préparation des données de validation si fournies
        validation_data_for_keras = None
        monitor_metric = "loss"

        if validation_data is not None:
            X_val_raw, y_val_raw = validation_data
            X_val_scaled = self.scaler.transform(X_val_raw)
            validation_data_for_keras = (X_val_scaled, y_val_raw)

            if weights_val is not None:
                validation_data_for_keras = (X_val_scaled, y_val_raw, weights_val)

            monitor_metric = "val_loss"

        # --- FILET DE SÉCURITÉ OPTIMISÉ ---
        early_stopper = EarlyStopping(
            monitor=monitor_metric,
            patience=7,  # Réduit de 15 à 7 : on coupe beaucoup plus vite si ça stagne !
            restore_best_weights=True,  # Indispensable pour rejeter les époques qui ont overfitté
            verbose=1,
        )

        # --- LANCEMENT DE L'ENTRAÎNEMENT ---
        history = self.model.fit(
            X_train_scaled,
            y_train,
            sample_weight=weights_train,
            validation_data=validation_data_for_keras,
            epochs=100,
            batch_size=2048,
            verbose=2,
            callbacks=[early_stopper],
        )
        return history

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data, batch_size=2048).flatten().ravel()
