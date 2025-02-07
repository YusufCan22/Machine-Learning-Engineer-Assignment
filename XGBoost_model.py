import pandas as pd
import numpy as np
import ta  # Teknik analiz kütüphanesi
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

class XGBoostStockPredictor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()  # Fiyat ölçekleme
        self.feature_scaler = MinMaxScaler()  # Teknik göstergeler

    def load_data(self, file_path):
        df = pd.read_csv(file_path, parse_dates=["Date"], sep=",", header=0, engine='python')

        # Beklenen sütun isimlerini belirleme
        expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    
        # Eğer sütun isimleri farklıysa hata ver
        if not all(col in df.columns for col in expected_columns):
           raise ValueError(f"Beklenen sütun isimleri: {expected_columns}, ancak dosyada farklı sütunlar var: {df.columns.tolist()}")

        df = df[expected_columns]
    
        # Tarihe göre sıralama
        df = df.sort_values(by="Date").reset_index(drop=True)

        print(f"✅ Veri Yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun.")
        return df


    def feature_engineering(self, df):
        """Teknik ve istatistiksel göstergeleri hesaplar."""

        #  Hareketli Medyan (Rolling Median) (Son 10 Günlük)
        df["Rolling_Median_10"] = df["Close"].rolling(window=10).median()

        #  Hareketli Standart Sapma (Rolling Std) 
        df["Rolling_Std_10"] = df["Close"].rolling(window=10).std()
        df["Rolling_Std_20"] = df["Close"].rolling(window=20).std()

        #  Çeyreklik (Quartiles) Hesaplamaları 
        df["Q1_10"] = df["Close"].rolling(window=10).quantile(0.25)
        df["Q3_10"] = df["Close"].rolling(window=10).quantile(0.75)

        #  Hareketli Varyans 
        df["Rolling_Var_10"] = df["Close"].rolling(window=10).var()

        #  Z-Score Hesaplama 
        df["Z_Score"] = (df["Close"] - df["Close"].rolling(window=10).mean()) / df["Rolling_Std_10"]

        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["Volatility"] = df["Close"].rolling(window=10).std()

        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["Bollinger_H"] = bb.bollinger_hband()
        df["Bollinger_L"] = bb.bollinger_lband()

        df["Momentum"] = df["Close"] - df["Close"].shift(4)
        df["ROC"] = ((df["Close"] - df["Close"].shift(4)) / df["Close"].shift(4)) * 100
        df["Diff_Close"] = df["Close"].diff()
        df["Price_Change"] = df["Close"].pct_change()

        # Hareketli geçmiş kapanış fiyatları
        df["Prev_Close_1"] = df["Close"].shift(1)
        df["Prev_Close_3"] = df["Close"].shift(3)
        df["Prev_Close_5"] = df["Close"].shift(5)
        df["Prev_Close_10"] = df["Close"].shift(10)
        df = df.dropna()  # Eksik verileri temizleme
        return df

    def preprocess_data(self, df):
        """Verileri ölçeklendirir."""
        important_features = [
        "Close", "Volume",  
        "SMA_10", "SMA_50", "SMA_200",  
        "MACD", "MACD_Signal",  
        "Bollinger_H", "Bollinger_L",  
        "Momentum", "ROC", "Diff_Close", "Price_Change",
    
        "Rolling_Median_10", "Rolling_Std_10", "Rolling_Std_20",
        "Q1_10", "Q3_10", "Rolling_Var_10", "Z_Score",
    
        # Hareketli geçmiş kapanış fiyatları
        "Prev_Close_1", "Prev_Close_3", "Prev_Close_5", "Prev_Close_10"
        ]

        df_features = df[important_features].astype(np.float32) 

        # Close fiyatını ayrı ölçeklendir
        df["Close"] = self.price_scaler.fit_transform(df[["Close"]])
        scaled_features = self.feature_scaler.fit_transform(df_features)

        return scaled_features, df["Close"].values

    def train_xgboost(self, X_train, y_train):
        """XGBoost modelini eğitir."""
        model = xgb.XGBRegressor(
            tree_method="hist",
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.5
        )

        model.fit(X_train, y_train)

        return model

if __name__ == "__main__":
    predictor = XGBoostStockPredictor()

    file_path = "stock.csv"  
    df = predictor.load_data(file_path)

    df = predictor.feature_engineering(df)
    scaled_data, y_scaled = predictor.preprocess_data(df)

    train_size = int(len(scaled_data) * 0.8)
    X_train, X_test = scaled_data[:train_size], scaled_data[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # XGBoost Eğitimi
    xgb_model = predictor.train_xgboost(X_train, y_train)

    # Test setinin tarihlerini al
    test_dates = df["Date"].iloc[train_size:].reset_index(drop=True)

    y_pred_xgb = xgb_model.predict(X_test)

    # Gerçek fiyat ölçeğinde tahminleri geri dönüştür
    y_pred_real = predictor.price_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1))
    y_test_real = predictor.price_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Performans Metrikleri
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    print(f"✅ R-squared (R²): {r2:.2f}")

    print(f"\n📊 XGBoost Sonuçları:")
    print(f"✅ MAE: {mae_real:.4f}")
    print(f"✅ RMSE: {rmse_real:.4f}")

    # Tahmin Grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_real, label="Gerçek Değerler", color="blue")
    plt.plot(test_dates, y_pred_real, label="XGBoost Tahminleri", color="red", linewidth=1.5)
    
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  
    
    plt.xlabel("Yıl", fontsize=12)
    plt.ylabel("Kapanış Fiyatı", fontsize=12)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=45)
    plt.title(f"XGBoost - Gerçek vs Tahmin\nMAE: {mae_real:.4f}, RMSE: {rmse_real:.4f}", fontsize=14, fontweight='bold')
    plt.show()

