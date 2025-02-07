import pandas as pd
import numpy as np
import ta  # Teknik analiz kütüphanesi
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class StockPredictor:
    def __init__(self, sequence_length=15, train_split=0.8):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.scaler = MinMaxScaler()

    def load_data(self, file_path):
        """Tek bir CSV dosyasını yükler ve hazırlar."""
        df = pd.read_csv(file_path, parse_dates=["Date"], sep=",", header=0, engine='python')
        
        # Sütun isimlerini kontrol etme
        expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if df.columns.tolist()[:7] != expected_columns:
            raise ValueError(f"Beklenen sütun isimleri: {expected_columns}, ancak dosyada farklı sütunlar var.")

        df.columns = expected_columns  
        df = df.sort_values(by="Date")  # Tarihe göre sıralama
        print(f"✅ Veri Yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun.")
        return df

    def feature_engineering(self, df):
        """Teknik göstergeleri hesaplar."""
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["Volatility"] = df["Close"].rolling(window=10).std()
        
        if "RSI" not in df.columns:
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["Bollinger_H"] = bb.bollinger_hband()
        df["Bollinger_L"] = bb.bollinger_lband()

        df = df.dropna()
        return df

    def preprocess_data(self, df):
        """Verileri ölçeklendirir ve dizilere dönüştürür."""
        df = self.feature_engineering(df)
        df = df[["Open", "High", "Low", "Volume", "Close", "SMA_10", "EMA_10", "Volatility", "RSI", "MACD", "MACD_Signal", "Bollinger_H", "Bollinger_L"]]
        
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data
    
    def create_sequences(self, data):
        """Verileri LSTM modeli için dizilere çevirir."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])  # Tüm özellikleri kullan
            y.append(data[i + self.sequence_length, -1])  
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """LSTM modeli oluşturur."""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            BatchNormalization(),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
if __name__ == "__main__":
    predictor = StockPredictor()
    
    file_path = "stock.csv"  
    df = predictor.load_data(file_path)
    
    scaled_data = predictor.preprocess_data(df)
    X, y = predictor.create_sequences(scaled_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    predictor.model = predictor.build_model((predictor.sequence_length, X_train.shape[2]))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = predictor.model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=512, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

    # Model tahminleri
    y_pred = predictor.model.predict(X_test)

    # Ters ölçeklendirme işlemi
    y_test_actual = predictor.scaler.inverse_transform(np.hstack((np.zeros((len(y_test), X_train.shape[2] - 1)), y_test.reshape(-1, 1))))[:, -1]
    y_pred_actual = predictor.scaler.inverse_transform(np.hstack((np.zeros((len(y_pred), X_train.shape[2] - 1)), y_pred.reshape(-1, 1))))[:, -1]

    # MAE & RMSE Hesaplamaları
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

    print(f"📉 Model Performans Değerleri:")
    print(f"✅ MAE (Ortalama Mutlak Hata): {mae:.2f}")
    print(f"✅ RMSE (Karekök Ortalama Hata): {rmse:.2f}")

    r2 = r2_score(y_test_actual, y_pred_actual)
    print(f"✅ R-squared (R²): {r2:.2f}")

    # Test verisinin tarihlerini al (Son len(y_test) kadar)
    test_dates = df["Date"].iloc[-len(y_test):]  

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_actual, label="Gerçek Değerler", color='blue')
    plt.plot(test_dates, y_pred_actual, label="Tahminler", color='red')

    # X ekseni için tarih formatlaması
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  

    plt.xlabel("Yıl") 
    plt.ylabel("Kapanış Fiyatı")
    plt.title(f"LSTM Modeli - Gerçek vs Tahmin\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)  

    plt.show()
