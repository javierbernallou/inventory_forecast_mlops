from pathlib import Path
import pandas as pd
import numpy as np


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

STATE_FILTER = "CA" 



def load_data():
    """Carga los 3 CSVs originales usando Pathlib."""
    
    sales = pd.read_csv(RAW_DATA_PATH / "sales_train_validation.csv")
    calendar = pd.read_csv(RAW_DATA_PATH / "calendar.csv")
    prices = pd.read_csv(RAW_DATA_PATH / "sell_prices.csv")
    
    print(f"   -> Sales original: {sales.shape}")
    print(f"   -> Calendar original: {calendar.shape}")
    print(f"   -> Prices original: {prices.shape}")
    return sales, calendar, prices


def optimize_dtypes(df, dataset_name):
    """Optimiza tipos de datos reemplazando 'object' por 'category' y reduciendo numéricos.
    """
    print(f"Optimizando datos para: {dataset_name}")
    initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < 500:  
                df[col] = df[col].astype('category')
        
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
            
    final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"   -> Reducción RAM {dataset_name}: {initial_memory:.1f}MB a {final_memory:.1f}MB")
    return df


def melt_sales_dataframe(sales_df, state_to_keep=None):
    """Aplica el filtro por estado y transforma el formato Wide a Long."""
    
    if state_to_keep:
        print(f"   -> Filtrando registros solo para el estado: {state_to_keep}")
        sales_df = sales_df[sales_df["state_id"] == state_to_keep].copy()
    
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    sales_long = sales_df.melt(
        id_vars=id_columns,
        var_name="d",
        value_name="sales"
    )
    
    sales_long["sales"] = sales_long["sales"].astype(np.int16)
    sales_long["d"] = sales_long["d"].astype("category")
    
    print(f"   -> Shape del dataset tras Melt: {sales_long.shape}")
    return sales_long


def merge_datasets(sales_long, calendar, prices, state_to_keep=None):
    """Une consecutivamente el Calendario y los Precios de forma eficiente."""
    print("Ejecutando Merges")
    
    calendar = optimize_dtypes(calendar, "Calendar")
    
    if state_to_keep:
        prices = prices[prices["store_id"].str.startswith(state_to_keep)].copy()
    prices = optimize_dtypes(prices, "Prices")

    print("   -> Cruzando Ventas con Calendario...")
    df = sales_long.merge(calendar, on="d", how="left")
    
    print("   -> Cruzando con Precios Históricos...")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    df = optimize_dtypes(df, "Dataset Final Integrado")
    return df


def save_processed_data(df):
    """Persiste el dataset resultante en formato Parquet optimizado."""
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    output_file = PROCESSED_DATA_PATH / "retail_dataset.parquet"
    
    df.to_parquet(output_file, index=False, compression="snappy")
    print(f"Archivo guardado")


def main():
    
    sales, calendar, prices = load_data()
    print("-" * 50)
    
    sales_long = melt_sales_dataframe(sales, state_to_keep=STATE_FILTER)
    print("-" * 50)
    
    final_df = merge_datasets(sales_long, calendar, prices, state_to_keep=STATE_FILTER)
    print("-" * 50)
    
    save_processed_data(final_df)
    print("Terminado.")


if __name__ == "__main__":
    main()