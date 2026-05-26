from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np

app = FastAPI(
    title="Walmart M5 Inventory Optimization API",
    description="API de producción para consultar predicciones de demanda y stock optimizado.",
    version="1.0.0"
)

PROCESSED_DATA_PATH = Path("data/processed")
DATA_FILE = PROCESSED_DATA_PATH / "optimized_inventory.parquet"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"No se encuentra el archivo {DATA_FILE}.")

inventory_db = pd.read_parquet(DATA_FILE)
inventory_db["store_id"] = inventory_db["store_id"].astype(str)
inventory_db["item_id"] = inventory_db["item_id"].astype(str)


@app.get("/")
def read_root():
    """Endpoint de bienvenida para verificar que la API está viva."""
    return {
        "status": "online",
        "project": "M5 Inventory Forecasting & Decision Optimization System",
        "version": "1.0.0"
    }


@app.get("/predict/{store_id}/{item_id}")
def get_inventory_recommendation(store_id: str, item_id: str):
    """Devuelve el forecast y el inventario optimizado para una tienda y producto específicos."""
    
    # Filtrar en nuestra base de datos optimizada (buscamos el último registro disponible)
    result = inventory_db[
        (inventory_db["store_id"] == store_id) & 
        (inventory_db["item_id"] == item_id)
    ]
    
    # Si no encontramos el producto o la tienda, lanzamos un error 404 de HTTP
    if result.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontraron datos para la combinación Store: '{store_id}' e Item: '{item_id}'."
        )
    
    # Tomamos el registro más reciente (última fila)
    latest_record = result.iloc[-1]
    
    # Construimos la respuesta de negocio
    return {
        "store_id": store_id,
        "item_id": item_id,
        "date_consulted": str(latest_record["date"]),
        "historical_sales_sample": float(latest_record["sales"]),
        "metrics": {
            "predicted_sales_next_day": round(float(latest_record["predicted_sales"]), 2),
            "optimized_inventory_to_stock": int(latest_record["optimized_inventory"])
        },
        "business_action": f"El sistema recomienda mantener {int(latest_record['optimized_inventory'])} unidades en stock."
    }