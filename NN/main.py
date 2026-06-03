from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    Class implementing a sequential neural network classifier with Keras.
    Optimized for the HiggsML dataset with batch acceleration.
    """

    def _init_(self, train_data):
        self.model = Sequential()

        # Nombre de variables d'entrée (features physiques)
        n_dim = train_data.shape[1]

        # 1. Couche d'entrée + Première couche cachée
        self.model.add(Dense(64, input_dim=n_dim, activation="relu"))
        self.model.add(Dropout(0.2))

        # 2. Deuxième couche cachée
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.2))

        # 3. Troisième couche cachée
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.2))

        # 4. Quatrième couche cachée (transition douce)
        self.model.add(Dense(32, activation="relu"))

        # 5. Couche de sortie (Classification binaire : Signal=1 / Background=0)
        self.model.add(Dense(1, activation="sigmoid"))

        # Compilation avec un Learning Rate de 0.002 adapté au grand batch
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.002),
            metrics=["accuracy"],
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):
        # Normalisation des données d'entraînement
        X_train = self.scaler.fit_transform(train_data)

        # Entraînement accéléré fixé à 5 époques
        self.model.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            epochs=30,  # Maintenu à 5 selon votre choix
            batch_size=2048,
            verbose=2,
        )

    def predict(self, test_data):
        # Application des paramètres de standardisation calculés sur le Train set
        test_data = self.scaler.transform(test_data)

        # Prédiction accélérée également par lot
        return (
            self.model.predict(test_data, batch_size=2048).flatten().ravel()
        )