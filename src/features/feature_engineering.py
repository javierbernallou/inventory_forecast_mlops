from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DATA_PATH = Path("data/processed")

def calendar_features(df):
    """Crear características temporales basadas en la fecha."""
    df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["weekday"] = df["date"].dt.weekday.astype(np.int8)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)


    return df

def lag_features(df):
    """Crear características del pasado para las ventas."""

    df = df.sort_values(by=["store_id", "item_id", "date"]).reset_index(drop=True)

    grupos = df.groupby(["store_id", "item_id"])

    #Para calcular la media movil de x días, shift() para evitar calcular la media de si mismo
    df["lag_1"] = grupos["sales"].shift(1).astype(np.float32)
    df["lag_7"] = grupos["sales"].shift(7).astype(np.float32)
    df["lag_14"] = grupos["sales"].shift(14).astype(np.float32)
    df["lag_28"] = grupos["sales"].shift(28).astype(np.float32)

    return df

def rolling_features(df):
    """ Calcula métricas mmoviles sobre las ventas. """

    grupos = df.groupby(["store_id", "item_id"])["sales"]

    df["rolling_mean_7"] = grupos.transform(lambda x: x.shift(1).rolling(7).mean()).astype(np.float32)
    
    df["rolling_std_7"] = grupos.transform(lambda x: x.shift(1).rolling(7).std()).astype(np.float32)
    
    df["rolling_mean_28"] = grupos.transform(lambda x: x.shift(1).rolling(28).mean()).astype(np.float32)
    
    return df


def main():
    input_file = PROCESSED_DATA_PATH / "retail_dataset.parquet"
    output_file = PROCESSED_DATA_PATH / "features_dataset.parquet"

    df = pd.read_parquet(input_file)
    
    df = calendar_features(df)
    df = lag_features(df)
    df = rolling_features(df)
    

    
    df.to_parquet(output_file, index=False, compression="snappy")
    print(f"Terminado.")


if __name__ == "__main__":
    main()