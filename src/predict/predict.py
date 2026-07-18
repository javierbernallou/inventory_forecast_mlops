# predict.py
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("models")

FEATURES = [
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "sell_price", "month", "weekday", "is_weekend"
]

SERVICE_LEVEL_Z = 1.65  # ~95% de confianza 

def load_model():
    model = XGBRegressor()
    model.load_model(MODELS_PATH / "xgboost_baseline.json")
    return model

def generate_forecast():
    df = pd.read_parquet(PROCESSED_DATA_PATH / "features_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    model = load_model()
    df["predicted_sales"] = model.predict(df[FEATURES])
    df["predicted_sales"] = np.clip(df["predicted_sales"], 0, None)

    # Safety stock = demanda prevista + buffer basado en la variabilidad histórica 
    # Cuanto más volátil la demanda, más stock de seguridad.
    df["recommended_stock"] = df["predicted_sales"] + SERVICE_LEVEL_Z * df["rolling_std_7"].fillna(0)
    df["recommended_stock"] = np.ceil(df["recommended_stock"]).astype(int)

    output_cols = ["date", "store_id", "item_id", "sales", "predicted_sales", "recommended_stock", "sell_price"]
    output_file = PROCESSED_DATA_PATH / "forecast_results.parquet"
    df[output_cols].to_parquet(output_file, index=False)
    print(f"Forecast guardado en {output_file} ({df.shape[0]} filas)")

if __name__ == "__main__":
    generate_forecast()