from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

QWEN_DIR = MODELS_DIR / "qwen"
LLAMA_DIR = MODELS_DIR / "llama"


