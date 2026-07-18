
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STATE_FILTER = os.getenv("STATE_FILTER", "CA")
 
# --- Rutas de datos ---
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
MODELS_PATH = Path(os.getenv("MODELS_PATH", "models"))
 
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "M5_Inventory_Forecasting")
 

HOLDING_COST_RATE = float(os.getenv("HOLDING_COST_RATE", "0.02"))
LOST_MARGIN_RATE = float(os.getenv("LOST_MARGIN_RATE", "0.30"))