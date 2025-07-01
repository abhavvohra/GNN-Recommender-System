import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class RecommenderTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, ratings_df, test_size=0.2, val_size=0.1):
        train_val, test = train_test_split(ratings_df, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        return train, val, test
    
    def create_dataloaders(self, train_df, val_df, batch_size=1024):
        train_users = torch.LongTensor(train_df['user_id'].values)
        train_items = torch.LongTensor(train_df['item_id'].values)
        train_ratings = torch.FloatTensor(train_df['rating'].values)
        
        val_users = torch.LongTensor(val_df['user_id'].values)
        val_items = torch.LongTensor(val_df['item_id'].values)
        val_ratings = torch.FloatTensor(val_df['rating'].values)
        
        train_dataset = TensorDataset(train_users, train_items, train_ratings)
        val_dataset = TensorDataset(val_users, val_items, val_ratings)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, data, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_users, batch_items, batch_ratings in tqdm(train_loader, desc="Training"):
            batch_users = batch_users.to(self.device)
            batch_items = batch_items.to(self.device)
            batch_ratings = batch_ratings.to(self.device)
            
            optimizer.zero_grad()
            
            predictions = self.model(data, batch_users, batch_items)
            
            user_emb = self.model.user_embedding(batch_users)
            item_emb = self.model.item_embedding(batch_items)
            
            loss = criterion(predictions, batch_ratings, user_emb, item_emb)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, data, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_users, batch_items, batch_ratings in val_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                predictions = self.model(data, batch_users, batch_items)
                
                user_emb = self.model.user_embedding(batch_users)
                item_emb = self.model.item_embedding(batch_items)
                
                loss = criterion(predictions, batch_ratings, user_emb, item_emb)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_ratings.cpu().numpy())
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return total_loss / len(val_loader), rmse, mae
    
    def train(self, data, train_df, val_df, epochs=100, lr=0.001, batch_size=1024):
        train_loader, val_loader = self.create_dataloaders(train_df, val_df, batch_size)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = CollaborativeFilteringLoss(lambda_reg=0.01)
        
        data = data.to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print("Starting training...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(data, train_loader, optimizer, criterion)
            val_loss, val_rmse, val_mae = self.validate(data, val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training completed!")
        
    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

from gnn_model import CollaborativeFilteringLoss