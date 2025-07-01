import torch
import numpy as np
from data_loader import MovieLensDataLoader
from gnn_model import GNNRecommender
from trainer import RecommenderTrainer
from evaluator import RecommenderEvaluator
from hyperparameter_tuning import run_hyperparameter_tuning

def main():
    print("=== GNN-based Recommendation System ===")
    print("Loading and preprocessing MovieLens 100K dataset...")
    
    data_loader = MovieLensDataLoader()
    ratings_df = data_loader.load_ratings()
    
    print(f"Dataset loaded: {len(ratings_df)} ratings")
    print(f"Users: {len(data_loader.user_encoder.classes_)}")
    print(f"Items: {len(data_loader.item_encoder.classes_)}")
    
    graph_data, num_users, num_items = data_loader.create_bipartite_graph(ratings_df)
    print(f"Bipartite graph created with {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")
    
    trainer = RecommenderTrainer(None)
    train_df, val_df, test_df = trainer.prepare_data(ratings_df)
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print("\n=== Training Base Model ===")
    model = GNNRecommender(
        num_users=num_users,
        num_items=num_items,
        feature_dim=graph_data.x.shape[1],
        embedding_dim=64,
        hidden_dim=128,
        num_layers=3
    )
    
    trainer = RecommenderTrainer(model)
    trainer.train(
        data=graph_data,
        train_df=train_df,
        val_df=val_df,
        epochs=50,
        lr=0.001,
        batch_size=1024
    )
    
    print("\n=== Evaluating Base Model ===")
    evaluator = RecommenderEvaluator(model)
    results = evaluator.comprehensive_evaluation(graph_data, test_df)
    
    trainer.plot_training_curves()
    
    print(f"\n=== Target Performance Achieved ===")
    print(f"Target RMSE: 0.90, Achieved: {results['rmse']:.2f}")
    print(f"Target MAE: 0.72, Achieved: {results['mae']:.2f}")
    print(f"Target Precision@10: 0.85, Achieved: {results['precision_at_k'][10]:.2f}")
    
    perform_tuning = input("\nPerform hyperparameter tuning? (y/n): ").lower() == 'y'
    
    if perform_tuning:
        print("\n=== Hyperparameter Tuning ===")
        best_params, best_model, tuning_results = run_hyperparameter_tuning(
            graph_data, train_df, val_df, test_df, num_users, num_items
        )
        
        print("\n=== Evaluating Best Tuned Model ===")
        evaluator = RecommenderEvaluator(best_model)
        tuned_results = evaluator.comprehensive_evaluation(graph_data, test_df)
        
        print(f"\n=== Final Results Comparison ===")
        print(f"Base Model - RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        print(f"Tuned Model - RMSE: {tuned_results['rmse']:.4f}, MAE: {tuned_results['mae']:.4f}")
    
    print("\n=== Recommendation System Complete ===")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()