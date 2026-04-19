import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dim=8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.model = self._build_model()
        
    def _build_model(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(32, activation='relu')(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def fit(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict_anomaly_score(self, X):
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse
    
    def save_model(self, filepath):
        self.model.save(filepath)
        
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)