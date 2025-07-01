import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class RecommenderEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def evaluate_ratings(self, data, test_df, batch_size=1024):
        test_users = torch.LongTensor(test_df['user_id'].values)
        test_items = torch.LongTensor(test_df['item_id'].values)
        test_ratings = torch.FloatTensor(test_df['rating'].values)
        
        test_dataset = TensorDataset(test_users, test_items, test_ratings)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        data = data.to(self.device)
        
        with torch.no_grad():
            for batch_users, batch_items, batch_ratings in test_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                predictions = self.model(data, batch_users, batch_items)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_ratings.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return rmse, mae, all_predictions, all_targets
    
    def precision_at_k(self, data, test_df, k=10, threshold=4.0):
        user_groups = test_df.groupby('user_id')
        precisions = []
        
        data = data.to(self.device)
        
        for user_id, group in user_groups:
            if len(group) < k:
                continue
                
            try:
                user_tensor = torch.LongTensor([user_id] * len(group)).to(self.device)
                item_tensor = torch.LongTensor(group['item_id'].values).to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(data, user_tensor, item_tensor)
                    predictions = predictions.cpu().numpy()
                
                actual_ratings = group['rating'].values
                
                pred_df = pd.DataFrame({
                    'item_id': group['item_id'].values,
                    'predicted_rating': predictions,
                    'actual_rating': actual_ratings
                })
                
                top_k_items = pred_df.nlargest(k, 'predicted_rating')
                
                if len(top_k_items) == 0:
                    continue
                    
                relevant_items = len(top_k_items[top_k_items['actual_rating'] >= threshold])
                precision = relevant_items / k
                precisions.append(precision)
                
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self, data, test_df, k=10, threshold=4.0):
        user_groups = test_df.groupby('user_id')
        recalls = []
        
        data = data.to(self.device)
        
        for user_id, group in user_groups:
            if len(group) < k:
                continue
                
            try:
                user_tensor = torch.LongTensor([user_id] * len(group)).to(self.device)
                item_tensor = torch.LongTensor(group['item_id'].values).to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(data, user_tensor, item_tensor)
                    predictions = predictions.cpu().numpy()
                
                actual_ratings = group['rating'].values
                
                pred_df = pd.DataFrame({
                    'item_id': group['item_id'].values,
                    'predicted_rating': predictions,
                    'actual_rating': actual_ratings
                })
                
                top_k_items = pred_df.nlargest(k, 'predicted_rating')
                total_relevant = len(group[group['rating'] >= threshold])
                
                if total_relevant == 0 or len(top_k_items) == 0:
                    continue
                    
                relevant_in_top_k = len(top_k_items[top_k_items['actual_rating'] >= threshold])
                recall = relevant_in_top_k / total_relevant
                recalls.append(recall)
                
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
        
        return np.mean(recalls) if recalls else 0.0
    
    def ndcg_at_k(self, data, test_df, k=10):
        user_groups = test_df.groupby('user_id')
        ndcgs = []
        
        data = data.to(self.device)
        
        for user_id, group in user_groups:
            if len(group) < k:
                continue
                
            try:
                user_tensor = torch.LongTensor([user_id] * len(group)).to(self.device)
                item_tensor = torch.LongTensor(group['item_id'].values).to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(data, user_tensor, item_tensor)
                    predictions = predictions.cpu().numpy()
                
                actual_ratings = group['rating'].values
                
                pred_df = pd.DataFrame({
                    'item_id': group['item_id'].values,
                    'predicted_rating': predictions,
                    'actual_rating': actual_ratings
                })
                
                top_k_items = pred_df.nlargest(k, 'predicted_rating')
                
                if len(top_k_items) == 0:
                    continue
                
                dcg = 0
                for i, (_, row) in enumerate(top_k_items.iterrows()):
                    dcg += (2 ** row['actual_rating'] - 1) / np.log2(i + 2)
                
                ideal_ratings = sorted(actual_ratings, reverse=True)[:k]
                idcg = 0
                for i, rating in enumerate(ideal_ratings):
                    idcg += (2 ** rating - 1) / np.log2(i + 2)
                
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)
                    
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def comprehensive_evaluation(self, data, test_df, k_values=[5, 10, 20]):
        print("=== Comprehensive Evaluation ===")
        
        rmse, mae, predictions, targets = self.evaluate_ratings(data, test_df)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        for k in k_values:
            precision_k = self.precision_at_k(data, test_df, k=k)
            recall_k = self.recall_at_k(data, test_df, k=k)
            ndcg_k = self.ndcg_at_k(data, test_df, k=k)
            
            print(f"\nK = {k}:")
            print(f"  Precision@{k}: {precision_k:.4f}")
            print(f"  Recall@{k}: {recall_k:.4f}")
            print(f"  NDCG@{k}: {ndcg_k:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'precision_at_k': {k: self.precision_at_k(data, test_df, k=k) for k in k_values},
            'recall_at_k': {k: self.recall_at_k(data, test_df, k=k) for k in k_values},
            'ndcg_at_k': {k: self.ndcg_at_k(data, test_df, k=k) for k in k_values}
        }