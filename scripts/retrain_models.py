import os
import shutil
import logging
from src.config.settings import MODELS_DIR
from scripts.train_all_models import main as train_all_models_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_models():
    if os.path.exists(MODELS_DIR):
        logger.info(f"Removing all models in {MODELS_DIR}...")
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    remove_models()
    logger.info("Retraining all models...")
    train_all_models_main()
    logger.info("Retraining complete!")

if __name__ == "__main__":
    main() 