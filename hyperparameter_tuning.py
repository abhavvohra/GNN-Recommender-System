import itertools
import torch
from gnn_model import GNNRecommender
from trainer import RecommenderTrainer
from evaluator import RecommenderEvaluator
import numpy as np

class HyperparameterTuner:
    def __init__(self, data, train_df, val_df, test_df, num_users, num_items):
        self.data = data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.num_users = num_users
        self.num_items = num_items
        self.results = []
        
    def tune(self, param_grid, max_trials=20):
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        if len(all_combinations) > max_trials:
            selected_combinations = np.random.choice(
                len(all_combinations), 
                size=max_trials, 
                replace=False
            )
            combinations_to_try = [all_combinations[i] for i in selected_combinations]
        else:
            combinations_to_try = all_combinations
        
        print(f"Trying {len(combinations_to_try)} parameter combinations...")
        
        best_rmse = float('inf')
        best_params = None
        best_model = None
        
        for i, param_combination in enumerate(combinations_to_try):
            params = dict(zip(param_names, param_combination))
            print(f"\nTrial {i+1}/{len(combinations_to_try)}: {params}")
            
            try:
                model = GNNRecommender(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    feature_dim=self.data.x.shape[1],
                    embedding_dim=params.get('embedding_dim', 64),
                    hidden_dim=params.get('hidden_dim', 128),
                    num_layers=params.get('num_layers', 3)
                )
                
                trainer = RecommenderTrainer(model)
                trainer.train(
                    data=self.data,
                    train_df=self.train_df,
                    val_df=self.val_df,
                    epochs=params.get('epochs', 50),
                    lr=params.get('lr', 0.001),
                    batch_size=params.get('batch_size', 1024)
                )
                
                evaluator = RecommenderEvaluator(model)
                rmse, mae, _, _ = evaluator.evaluate_ratings(self.data, self.val_df)
                precision_10 = evaluator.precision_at_k(self.data, self.val_df, k=10)
                
                result = {
                    'params': params,
                    'rmse': rmse,
                    'mae': mae,
                    'precision_10': precision_10
                }
                
                self.results.append(result)
                
                print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Precision@10: {precision_10:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
                    best_model = model
                    torch.save(model.state_dict(), 'best_tuned_model.pth')
                    print("New best model saved!")
                
            except Exception as e:
                print(f"Failed with error: {e}")
                continue
        
        print(f"\n=== Hyperparameter Tuning Complete ===")
        print(f"Best RMSE: {best_rmse:.4f}")
        print(f"Best parameters: {best_params}")
        
        return best_params, best_model, self.results
    
    def get_results_summary(self):
        if not self.results:
            print("No results available. Run tuning first.")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['rmse'])
        
        print("\n=== Top 5 Results ===")
        for i, result in enumerate(sorted_results[:5]):
            print(f"\nRank {i+1}:")
            print(f"  Parameters: {result['params']}")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  MAE: {result['mae']:.4f}")
            print(f"  Precision@10: {result['precision_10']:.4f}")
        
        return sorted_results

def run_hyperparameter_tuning(data, train_df, val_df, test_df, num_users, num_items):
    param_grid = {
        'embedding_dim': [32, 64, 128],
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'lr': [0.0005, 0.001, 0.002],
        'batch_size': [512, 1024],
        'epochs': [30, 50]
    }
    
    tuner = HyperparameterTuner(data, train_df, val_df, test_df, num_users, num_items)
    best_params, best_model, results = tuner.tune(param_grid, max_trials=15)
    tuner.get_results_summary()
    
    return best_params, best_model, results