# Stock Price Prediction

## Project Overview
This project aims to predict stock price movements using machine learning models, specifically LSTM (Long Short-Term Memory) and XGBoost (Extreme Gradient Boosting). The models are trained to predict the next day's closing price based on historical stock data and various technical indicators.

## Project Structure
```
stock-price-prediction
 ┣ data                  # Folder containing stock CSV files
 ┣ LSTM_model.py         # LSTM model implementation
 ┣ XGBoost_model.py      # XGBoost model implementation
 ┣ report.pdf            # Detailed project report
 ┣ README.md             # This documentation file
```

## Dataset
- The dataset contains historical stock market data from Kaggle.
- Each stock file includes: Date, Open, High, Low, Close, Adjusted Close, and Volume.

## Features & Technical Indicators
The models utilize a variety of features, including:
- Moving Averages: SMA (10, 50, 200), EMA (10)
- Momentum Indicators: RSI, MACD, MACD Signal, ROC
- Volatility Measures: Bollinger Bands, Rolling Standard Deviation
- Statistical Features: Rolling Median, Quartiles, Z-Score
- Historical Price Data: Previous close prices (1, 3, 5, 10 days)

## Models Implemented
### LSTM (Deep Learning)
- Bidirectional LSTM layers for better sequential learning.
- Batch Normalization & Dropout to improve stability and prevent overfitting.
- Trained using Adam optimizer.

### XGBoost (Tree-Based Model)
- Gradient Boosting Trees to capture complex relationships.
- Feature engineering focused on statistical patterns.
- Optimized with fine-tuned hyperparameters.

## Installation & Usage
### Step 1: Access the Repository  
[Machine Learning Engineer Assignment Repository](https://github.com/YusufCan22/Machine-Learning-Engineer-Assignment)

### Step 2: Install Dependencies
pandas
numpy
ta
scikit-learn
xgboost
tensorflow
matplotlib
joblib

### Step 3: Run the Models
#### Running LSTM Model
```bash
python LSTM_model.py
```
#### Running XGBoost Model
```bash
python XGBoost_model.py
```

---
Author: Yusuf Can Gültekin  
GitHub: https://github.com/YusufCan22/Machine-Learning-Engineer-Assignment.git
Email: ycgultekin1@gmail.com
Version : 1.0.0

