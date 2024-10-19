import os
import logging
from datetime import datetime
class Config:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_path = os.path.join(self.base_path, 'data', 'raw', 'videos')
        self.processed_data_path = os.path.join(self.base_path, 'data', 'processed', 'keypoints')
        self.model_path = os.path.join(self.base_path, 'models', 'saved_models', 'action_model.keras')
        self.log_path = os.path.join(self.base_path, 'models', 'logs', 'app.log')

        # Ensure directories exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # List of actions (i.e., directories in the raw videos folder)
        self.actions = [d for d in os.listdir(self.raw_data_path) if os.path.isdir(os.path.join(self.raw_data_path, d))]

    def get_addon_path(self):
        return self.base_path
    
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
