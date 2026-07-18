from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

from config import PROCESSED_DATA_PATH, HOLDING_COST_RATE, LOST_MARGIN_RATE


def compute_critical_ratio(sell_price, holding_cost_rate=HOLDING_COST_RATE, lost_margin_rate=LOST_MARGIN_RATE):
    """Calcula el ratio crítico del modelo newsvendor para cada producto.

    Co (overage cost): coste de tener una unidad de más en stock.
    La aproximamos como un % del precio de venta (capital inmovilizado,
    espacio de almacén, riesgo de obsolescencia). Por defecto, 2% del precio.

    Cu (underage cost): coste de no tener una unidad cuando se necesita.
    La aproximamos como el margen perdido en esa venta: % del precio de
    venta que representa beneficio. Por defecto, 30% del precio.

    Ambos costes se expresan en función del sell_price para que el ratio
    sea independiente de la escala de precios entre productos: un producto
    de 1€ y uno de 100€ pueden tener la misma economía relativa de
    sobre-stock vs. rotura de stock.
    """
    overage_cost = holding_cost_rate * sell_price
    underage_cost = lost_margin_rate * sell_price

    # Ratio crítico: la pieza central del modelo newsvendor. Cuanto más caro
    # es quedarse corto (Cu) respecto a quedarse largo (Co), más cerca de 1
    # está este ratio, y más colchón de seguridad recomendará el modelo.
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    return critical_ratio


def optimize_inventory_vectorized(df):
    critical_ratio = compute_critical_ratio(df["sell_price"])

    z = norm.ppf(critical_ratio)

    demand_std = df["rolling_std_7"].fillna(0)

  
    optimized_inventory = df["predicted_sales"] + z * demand_std

    optimized_inventory = np.clip(optimized_inventory, 0, None)
    optimized_inventory = np.ceil(optimized_inventory).astype(int)

    return optimized_inventory, critical_ratio, z


def run_optimization_pipeline():
    input_file = PROCESSED_DATA_PATH / "forecast_results.parquet"
    output_file = PROCESSED_DATA_PATH / "optimized_inventory.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"No se encuentra {input_file}. Ejecuta predict.py primero.")

    df = pd.read_parquet(input_file)
    print(f"   -> Dataset cargado con {df.shape[0]} filas.")

    optimized_inventory, critical_ratio, z = optimize_inventory_vectorized(df)
    df["optimized_inventory"] = optimized_inventory
    df["critical_ratio"] = critical_ratio.round(3)
    df["safety_factor_z"] = z.round(3)

    df.to_parquet(output_file, index=False)
    print(f"Guardado en: {output_file}")
    print(df[["date", "store_id", "item_id", "sales", "predicted_sales",
              "critical_ratio", "safety_factor_z", "optimized_inventory"]].head())


if __name__ == "__main__":
    run_optimization_pipeline()