
from fastapi import FastAPI
import requests
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

API_KEY = "594e51de581d4df69d4fb5433f31c8b5"

# Load pretrained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

def get_data(symbol="USD/JPY"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&apikey={API_KEY}&outputsize=50&format=JSON"
    r = requests.get(url).json()
    df = pd.DataFrame(r['values'])
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})
    return df

@app.get("/signal")
def get_signal():
    df = get_data()
    last = df.tail(1)[['open', 'high', 'low', 'close']].values[0]
    prob = model.predict_proba([last])[0][1]
    if prob >= 0.7:
        return {"signal": "BUY", "probability": prob}
    else:
        return {"signal": "NO TRADE", "probability": prob}
