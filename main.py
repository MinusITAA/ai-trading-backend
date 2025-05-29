
from fastapi import FastAPI
import requests
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

API_KEY = "594e51de581d4df69d4fb5433f31c8b5"

# Load pretrained model
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Errore nel caricamento modello: {e}")

def get_data(symbol="USD/JPY"):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&apikey={API_KEY}&outputsize=50&format=JSON"
        r = requests.get(url, timeout=10).json()
        if 'values' not in r:
            print("Valori non presenti nella risposta TwelveData")
            return None
        df = pd.DataFrame(r['values'])
        df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})
        return df
    except Exception as e:
        print(f"Errore nel recupero dati: {e}")
        return None

@app.get("/signal")
def get_signal():
    if model is None:
        return {"signal": "ERROR", "probability": 0, "reason": "Modello non caricato"}

    df = get_data()
    if df is None or df.empty:
        return {"signal": "ERROR", "probability": 0, "reason": "Dati non validi"}

    try:
        last = df.tail(1)[['open', 'high', 'low', 'close']].values[0]
        prob = model.predict_proba([last])[0][1]
        if prob >= 0.7:
            return {"signal": "BUY", "probability": float(prob)}
        else:
            return {"signal": "NO TRADE", "probability": float(prob)}
    except Exception as e:
        print(f"Errore nella previsione: {e}")
        return {"signal": "ERROR", "probability": 0, "reason": str(e)}
