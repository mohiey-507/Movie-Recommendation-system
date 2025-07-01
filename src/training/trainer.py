
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import get_device
from src.utils.logging import Logger

from typing import Dict, List, Optional, Callable


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Callable,
        device: torch.device = None,
        scheduler: Optional[_LRScheduler] = None,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        model_checkpoint_freq: int = 5,
        **kwargs
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or get_device()
        self.scheduler = scheduler
        self.model_checkpoint_freq = model_checkpoint_freq
        self.kwargs = kwargs
        
        # Initialize logger
        self.logger = Logger(log_dir=log_dir, experiment_name=experiment_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Track best metrics
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_rmse': float('inf'),
        }

    def _process_batch(self, batch):
        inputs = {
            'user_id': batch['user'].to(self.device),
            'item_id': batch['movie'].to(self.device)
        }
        targets = {
            'rating': batch['rating'].to(self.device)
        }
        return inputs, targets
    
    def compute_loss(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return 
    
    def train_epoch(self) -> Dict[str, float]:

        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            user_id = batch['user'].to(self.device)
            item_id = batch['movie'].to(self.device)
            rating = batch['rating'].to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(user_id, item_id)
            loss = self.criterion(predictions, rating)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:

        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                user_id = batch['user'].to(self.device)
                item_id = batch['movie'].to(self.device)
                rating = batch['rating'].to(self.device)
                
                predictions = self.model(user_id, item_id)
                loss = self.compute_loss(predictions, rating)
                
                mse = torch.mean((predictions - rating) ** 2)
                rmse = torch.sqrt(mse)
                
                total_loss += loss.item()
                total_rmse += rmse.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_rmse = total_rmse / len(self.val_loader)
        
        return {
            'val_loss': avg_loss,
            'val_rmse': avg_rmse
        }
    
    def fit(self, num_epochs: int, early_stopping: int = 5) -> Dict[str, List[float]]:
        history = {}
        no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            self.logger.logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            metrics = {**train_metrics, **val_metrics}
            
            self.logger.log_metrics(metrics, step=epoch)
            
            # Update best metrics and save model if improved
            improved = False
            for metric_name, current_value in metrics.items():
                if metric_name.startswith('val_') and current_value < self.best_metrics.get(metric_name, float('inf')):
                    self.best_metrics[metric_name] = current_value
                    improved = True
            
            if improved:
                self.logger.logger.info(f"Validation metrics improved. Saving model...")
                self.logger.save_model(self.model, 'best_model.pth')
                no_improve = 0
            else:
                no_improve += 1
                
                if epoch % self.model_checkpoint_freq == 0:
                    self.logger.save_model(self.model, f'model_epoch_{epoch}.pth')
                
                if no_improve >= early_stopping:
                    break
        
        self.logger.save_model(self.model, 'final_model.pth')
        
        self.logger.close()
        
        return history