"""Project configuration for Peshawar Green-Heat Mapper."""
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
ENV_FOUND = ENV_PATH.exists()
ENV_TXT_MISTAKE = (ENV_PATH.parent / ".env.txt").exists()  # Notepad often saves ".env" as ".env.txt"
try:  # load keys from .env if python-dotenv is installed (utf-8-sig tolerates Notepad's BOM)
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=True, encoding="utf-8-sig")
except ImportError:
    pass

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

# Peshawar AOI (approx central urban)
PESHAWAR_BBOX = [71.45, 33.95, 71.65, 34.15]  # lon_min, lat_min, lon_max, lat_max
PESHAWAR_CENTER = [34.05, 71.55]  # lat, lon for maps

# Sentinel-2 bands of interest
S2_BANDS = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

# NDVI threshold for vegetation baseline
NDVI_VEG_THRESHOLD = 0.3

# Resolution notes
S2_RES_M = 10
LANDSAT_LST_RES_M = 100  # approximate after processing

# Budget defaults for planner (arbitrary units, e.g. trees or $k)
DEFAULT_BUDGET = 500
DEFAULT_WATER_ACCESS_WEIGHT = 0.3

# Species list (vetted, local-ish for KP / Peshawar climate)
VETTED_SPECIES = [
    {"name": "Dalbergia sissoo (Shisham)", "water": "medium", "cooling": "high", "source": "KP Forest Dept guidance"},
    {"name": "Albizia lebbeck (Siris)", "water": "low", "cooling": "high", "source": "Canopy tree recommendations"},
    {"name": "Ficus religiosa (Peepal)", "water": "medium", "cooling": "high", "source": "Local canopy species"},
    {"name": "Azadirachta indica (Neem)", "water": "low", "cooling": "medium", "source": "Drought-tolerant urban tree"},
    {"name": "Melia azedarach (Bakain)", "water": "low", "cooling": "medium", "source": "KP recommended"},
]
