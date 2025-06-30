import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import torch

class Logger:
    def __init__(self, log_dir: str = "./logs", experiment_name: Optional[str] = None):

        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.experiment_dir = os.path.join(self.log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.experiment_dir, 'training.log')
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicate logs
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store metrics
        self.metrics: Dict[str, list] = {}
        
        self.logger.info(f"Logging to {self.log_file}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to both console and file.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch
        """
        # Update metrics history
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
        
        # Format metrics string
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                            for k, v in metrics.items())
        self.logger.info(f"Step {step} - {metrics_str}")
        
        # Save metrics to JSON file
        metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_config(self, config: Dict[str, Any]):
        config_file = os.path.join(self.experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {config_file}")
    
    def save_model(self, model: torch.nn.Module, filename: str):
        model_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(model_dir, filename)
        torch.save(model.state_dict(), checkpoint_path)
        self.logger.info(f"Model saved to {checkpoint_path}")
    
    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
