from pathlib import Path
import pandas as pd
import numpy as np
import pulp

PROCESSED_DATA_PATH = Path("data/processed")

def optimize_single_item(predicted_sales, recommended_stock, max_capacity=500):
    """Función matemática para optimizar el inventario de una sola fila.
    
    Busca minimizar el coste de almacenamiento respetando las restricciones físicas.
    """
    problem = pulp.LpProblem("SingleItemInventory", pulp.LpMinimize)
    
    # 2. Variable de decisión: Cuánto inventario físico pedir/mantener
    # El inventario no puede ser negativo (lowBound=0) ni superar la capacidad del almacén
    inventory = pulp.LpVariable(
        "inventory", 
        lowBound=0, 
        upBound=max_capacity, 
        cat="Integer"  # Forzamos a que decida en unidades enteras
    )
    
    # Asumimos un coste teórico de 1€ por unidad almacenada al día
    holding_cost_unit = 1.0
    problem += holding_cost_unit * inventory
    
    problem += inventory >= recommended_stock
    
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
    return int(inventory.varValue) if inventory.varValue is not None else int(recommended_stock)


def run_optimization_pipeline():
    input_file = PROCESSED_DATA_PATH / "forecast_results.parquet"
    output_file = PROCESSED_DATA_PATH / "optimized_inventory.parquet"
    
    if not input_file.exists():
        raise FileNotFoundError(f"No se encuentra {input_file}.")
        
    df = pd.read_parquet(input_file)
    
    print(f"   -> Dataset cargado con {df.shape[0]} filas.")
    
    df["optimized_inventory"] = df.apply(
        lambda row: optimize_single_item(row["predicted_sales"], row["recommended_stock"]), 
        axis=1
    )
    
    df.to_parquet(output_file, index=False)
    print(f"Guardado en: {output_file}")
    print(df[["date", "store_id", "item_id", "sales", "predicted_sales", "optimized_inventory"]].head())


if __name__ == "__main__":
    run_optimization_pipeline()